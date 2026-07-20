#!/usr/bin/env python3
"""agent-kit local worker: execute needs-local-run PR tasks on this machine.

Finds open PRs labeled `needs-local-run`, checks each branch out into a
temporary git worktree, runs the command configured in `.agent/config.json`
(read from the default branch, never from the PR branch), commits and pushes
produced artifacts back to the PR branch, comments a summary, and flips the
label to `local-run-done`.

Run from anywhere inside the target repo:

  python scripts/worker.py             # process the queue once
  python scripts/worker.py --loop 300  # poll every 5 minutes

Stdlib only; requires `git` and an authenticated `gh`.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK_FILE_RE = re.compile(r"^\.agent/tasks/issue-(\d+)\.json$")
OUTPUT_TAIL_LINES = 60
OUTPUT_TAIL_CHARS = 6000
LABEL_TODO = "needs-local-run"
LABEL_DONE = "local-run-done"


def sh(args, cwd=None, check=True, timeout=None):
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


def gh_json(args, cwd=None):
    return json.loads(sh(["gh", *args], cwd=cwd).stdout)


def tail(text, lines=OUTPUT_TAIL_LINES, chars=OUTPUT_TAIL_CHARS):
    out = "\n".join(text.splitlines()[-lines:])
    return out[-chars:]


def matching_artifacts(worktree, patterns):
    return {g: [str(p.relative_to(worktree)) for p in Path(worktree).glob(g)] for g in patterns}


def load_config(root, default_branch):
    """Read worker config from the default branch, never the working tree or
    a PR branch, so a PR cannot change what command runs."""
    sh(["git", "fetch", "origin", default_branch], cwd=root)
    show = sh(["git", "show", f"origin/{default_branch}:.agent/config.json"], cwd=root, check=False)
    if show.returncode != 0:
        sys.exit(f"no .agent/config.json on origin/{default_branch}; run agent-kit setup first")
    config = json.loads(show.stdout).get("worker", {})
    if not config.get("command"):
        sys.exit(
            f"worker.command is not set in .agent/config.json on "
            f"origin/{default_branch}; set it and push before running "
            "the worker"
        )
    return config


def find_task_file(pr_number):
    files = sh(["gh", "pr", "diff", str(pr_number), "--name-only"]).stdout.split()
    matches = [f for f in files if TASK_FILE_RE.match(f)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one .agent/tasks/issue-<n>.json in the PR diff, "
            f"found {matches or 'none'}"
        )
    return matches[0]


def comment(pr_number, body):
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        path = f.name
    sh(["gh", "pr", "comment", str(pr_number), "--body-file", path])
    Path(path).unlink()


def process_pr(root, config, pr):
    number, branch = pr["number"], pr["headRefName"]
    print(f"PR #{number} ({branch}): processing")
    task_path = find_task_file(number)
    sh(["git", "fetch", "origin", branch], cwd=root)
    worktree = tempfile.mkdtemp(prefix=f"agent-kit-pr{number}-")
    try:
        sh(["git", "worktree", "add", "--detach", worktree, f"origin/{branch}"], cwd=root)
        task = json.loads((Path(worktree) / task_path).read_text())
        artifacts = task.get("expected_artifacts", [])

        if artifacts and all(matching_artifacts(worktree, artifacts).values()):
            print(f"PR #{number}: all expected artifacts already present, relabeling only")
            relabel(number)
            return

        command = config["command"].replace("{task_file}", task_path)
        timeout = float(config.get("max_minutes", 120)) * 60
        print(f"PR #{number}: running `{command}` (timeout {timeout / 60:.0f}m)")
        start = time.monotonic()
        try:
            run = subprocess.run(
                command,
                shell=True,
                cwd=worktree,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            output = run.stdout + run.stderr
            failed = run.returncode != 0
            status = f"exit {run.returncode}"
        except subprocess.TimeoutExpired as e:
            output = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(
                errors="replace"
            )
            failed = True
            status = f"timed out after {config.get('max_minutes', 120)} minutes"
        duration = time.monotonic() - start

        missing = [g for g, hits in matching_artifacts(worktree, artifacts).items() if not hits]
        if not failed and missing:
            failed = True
            status += f"; missing expected artifacts: {', '.join(missing)}"

        changed = sh(["git", "status", "--porcelain", "-uall"], cwd=worktree).stdout
        pushed = []
        if not failed and changed:
            sh(["git", "add", "-A"], cwd=worktree)
            sh(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Local run results for issue #{task.get('issue', '?')} (PR #{number})",
                ],
                cwd=worktree,
            )
            sh(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], cwd=worktree)
            pushed = [line[3:] for line in changed.splitlines()]

        header = "Local run failed" if failed else "Local run complete"
        body = (
            f"### {header}\n\n"
            f"- command: `{command}`\n"
            f"- status: {status}\n"
            f"- duration: {duration / 60:.1f} minutes\n"
            f"- artifacts pushed: {', '.join(f'`{p}`' for p in pushed) or 'none'}\n\n"
            f"<details><summary>output tail</summary>\n\n"
            f"```\n{tail(output)}\n```\n\n</details>\n"
        )
        comment(number, body)
        if failed:
            print(f"PR #{number}: FAILED ({status}); label left as {LABEL_TODO}")
        else:
            relabel(number)
            print(f"PR #{number}: done ({status}, {len(pushed)} files pushed)")
    finally:
        sh(["git", "worktree", "remove", "--force", worktree], cwd=root, check=False)


def relabel(pr_number):
    sh(
        [
            "gh",
            "pr",
            "edit",
            str(pr_number),
            "--remove-label",
            LABEL_TODO,
            "--add-label",
            LABEL_DONE,
        ]
    )


def run_once(root):
    default_branch = gh_json(["repo", "view", "--json", "defaultBranchRef"], cwd=root)[
        "defaultBranchRef"
    ]["name"]
    config = load_config(root, default_branch)
    prs = gh_json(
        ["pr", "list", "--state", "open", "--label", LABEL_TODO, "--json", "number,headRefName"],
        cwd=root,
    )
    if not prs:
        print(f"no open PRs labeled {LABEL_TODO}")
    for pr in prs:
        try:
            process_pr(root, config, pr)
        except Exception as e:
            print(f"PR #{pr['number']}: error: {e}", file=sys.stderr)
            try:
                comment(pr["number"], f"### Local run failed\n\n```\n{e}\n```\n")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--loop",
        type=float,
        metavar="SECONDS",
        help="poll forever at this interval (default: run once)",
    )
    args = parser.parse_args()
    root = sh(["git", "rev-parse", "--show-toplevel"]).stdout.strip()
    while True:
        run_once(root)
        if args.loop is None:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""agent-kit drain: implement queued issues with local Claude Code agents.

For each open issue labeled `queued` (oldest first), drain:
  1. creates an isolated git worktree on branch claude/issue-<n>,
  2. runs a headless Claude Code session (`claude -p`) in it, which implements
     the issue per the CLAUDE.md contract, pushes the branch, and opens a PR,
  3. verifies the PR exists and flips the issue labels,
  4. removes the worktree.

Sessions run on this machine with your full toolchain and stream output here;
Ctrl-C aborts the current session. Configure the session (permission mode,
allowed tools, model) via "implementer.claude_args" in .agent/config.json.

  python scripts/drain.py                # implement one issue
  python scripts/drain.py --limit 3      # up to 3, sequentially
  python scripts/drain.py --interactive  # steerable session instead of headless
  python scripts/drain.py --dry-run      # list what would be dispatched
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_CLAUDE_ARGS = '--permission-mode acceptEdits --allowedTools "Bash(git:*)" "Bash(gh:*)"'
DEFAULT_MAX_MINUTES = 45

# Headless `claude -p` silently bills an inherited API key (e.g. one loaded via
# `uv run --env-file .env`) instead of the subscription login; strip the
# credential overrides so sessions always bill the subscription.
SESSION_ENV = {
    k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
}


def sh(args, cwd=None, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def gh_json(args):
    return json.loads(sh(["gh", *args]).stdout)


def implementer_config(root: Path) -> dict:
    path = root / ".agent" / "config.json"
    if path.exists():
        return json.loads(path.read_text()).get("implementer", {})
    return {}


def build_prompt(repo: str, number: int, branch: str, default_branch: str) -> str:
    return (
        f"You are an autonomous implementer for {repo}.\n"
        f"Implement GitHub issue #{number}: read it with `gh issue view {number} --repo {repo}`,"
        f" then follow the CLAUDE.md contract in the repo root.\n"
        f"You are in an isolated git worktree already on branch `{branch}`"
        f" (cut from origin/{default_branch}), so skip branch creation.\n"
        f"Implement the issue, run the repo's checks, commit, push with"
        f" `git push -u origin {branch}`, open a PR with `Closes #{number}` in the body,"
        f" and apply the labels the contract requires.\n"
        f"Do not merge the PR. If the issue cannot be implemented, comment on it explaining why."
    )


def drain_issue(root, repo, default_branch, issue, config, interactive) -> bool:
    number = issue["number"]
    branch = f"claude/issue-{number}"
    print(f"=== #{number}: {issue['title']}  ({branch})")
    sh(["git", "fetch", "origin", default_branch], cwd=root)
    worktree = tempfile.mkdtemp(prefix=f"agent-kit-issue{number}-")
    sh(["git", "worktree", "add", "-B", branch, worktree, f"origin/{default_branch}"], cwd=root)
    try:
        cmd = ["claude"]
        if not interactive:
            cmd.append("-p")
        cmd.append(build_prompt(repo, number, branch, default_branch))
        cmd += shlex.split(config.get("claude_args", DEFAULT_CLAUDE_ARGS))
        timeout = (
            None if interactive else float(config.get("max_minutes", DEFAULT_MAX_MINUTES)) * 60
        )
        start = time.monotonic()
        try:
            # No capture: the session streams to this terminal so you can watch it work.
            session = subprocess.run(cmd, cwd=worktree, timeout=timeout, env=SESSION_ENV)
        except subprocess.TimeoutExpired:
            print(f"--- session timed out after {timeout / 60:.0f}m", file=sys.stderr)
            return False
        print(f"--- session exit {session.returncode} after {(time.monotonic() - start) / 60:.1f}m")

        prs = gh_json(["pr", "list", "--repo", repo, "--head", branch, "--json", "number,url"])
        if not prs:
            print(f"--- no PR found for {branch}; issue stays queued", file=sys.stderr)
            return False
        print(f"--- PR opened: {prs[0]['url']}")
        sh(
            [
                "gh",
                "issue",
                "edit",
                str(number),
                "--repo",
                repo,
                "--remove-label",
                "queued",
                "--add-label",
                "in-progress",
            ],
            check=False,
        )
        return True
    finally:
        sh(["git", "worktree", "remove", "--force", worktree], cwd=root, check=False)
        # The branch lives on the remote (and its PR); drop the local ref.
        sh(["git", "branch", "-D", branch], cwd=root, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit", type=int, default=1, help="implement at most N issues (default 1)"
    )
    parser.add_argument("--dry-run", action="store_true", help="list issues without implementing")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="run each session interactively (steerable) instead of headless",
    )
    args = parser.parse_args()

    root = Path(sh(["git", "rev-parse", "--show-toplevel"]).stdout.strip())
    info = gh_json(["repo", "view", "--json", "nameWithOwner,defaultBranchRef"])
    repo = info["nameWithOwner"]
    default_branch = info["defaultBranchRef"]["name"]

    issues = [
        i
        for i in gh_json(
            [
                "issue",
                "list",
                "--repo",
                repo,
                "--state",
                "open",
                "--label",
                "queued",
                "--json",
                "number,title,labels",
            ]
        )
        if not any(label["name"] == "in-progress" for label in i["labels"])
    ]
    issues.sort(key=lambda i: i["number"])
    issues = issues[: args.limit]
    if not issues:
        print("queue is empty")
        return

    config = implementer_config(root)
    done = 0
    for issue in issues:
        if args.dry_run:
            print(f"would implement #{issue['number']}: {issue['title']}")
            continue
        done += drain_issue(root, repo, default_branch, issue, config, args.interactive)
    if not args.dry_run:
        print(f"\n{done}/{len(issues)} issue(s) now have PRs")


if __name__ == "__main__":
    main()

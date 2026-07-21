#!/usr/bin/env python3
"""Local-run entrypoint for the agent-kit worker (see .agent/config.json).

The worker invokes this with the path to the PR's .agent/tasks/issue-<n>.json.
The task file's `command_args` names one experiment directory under
experiments/runs/, and the sweep its config.py defines runs here on the
maintainer's machine. Whatever the sweep writes inside the experiment
directory (runs.jsonl; checkpoints stay gitignored via *.pt) is pushed back
to the PR branch by the worker.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

RUNS_ROOT = Path('experiments/runs')
LOCAL_DATA = ('input.txt',)  # gitignored corpus files sweeps read from the repo root


def ensure_local_data():
    """The worker runs this script in an ephemeral worktree, which starts
    without gitignored local data. Copy it from the main checkout the
    worktree hangs off (the parent of the shared git dir)."""
    git_dir = subprocess.run(['git', 'rev-parse', '--git-common-dir'],
                             capture_output=True, text=True, check=True).stdout.strip()
    main_root = Path(git_dir).resolve().parent
    for name in LOCAL_DATA:
        src, dst = main_root / name, Path(name)
        if src.is_file() and not dst.exists():
            print(f'copying {src} -> {dst}')
            shutil.copyfile(src, dst)


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: run_task.py <task_file.json>')
    task = json.loads(Path(sys.argv[1]).read_text())
    exp_dir = Path(str(task['command_args']).strip())
    if RUNS_ROOT.resolve() not in exp_dir.resolve().parents:
        sys.exit(f'command_args must name a directory under {RUNS_ROOT}/, got {exp_dir}')
    if exp_dir.name == '_template':
        sys.exit('refusing to run the template directory')
    if not (exp_dir / 'config.py').is_file():
        sys.exit(f'{exp_dir}/config.py not found')
    ensure_local_data()
    subprocess.run([sys.executable, '-m', 'experiments.sweep', str(exp_dir)],
                   check=True)


if __name__ == '__main__':
    main()

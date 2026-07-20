# Agent contract for kwu93/nanoGPT

This repo uses the agent-kit issue-queue workflow.
Issues labeled `agent-task` are units of work for coding agents.
This file is the contract every implementing agent must follow.

## Implementing an issue

When asked to implement issue N (usually by a local drain session; sometimes an `@claude implement` comment):

1. Read the issue: the goal, the acceptance criteria, and the optional "Local run" section.
2. Create branch `claude/issue-<n>` off `master` (drain sessions start already on this branch in a worktree; skip creation then).
3. Implement the goal.
   Keep changes minimal and focused on the issue.
4. If the issue has a "Local run" section, do NOT execute it here.
   CI runners lack GPU and local data; local-run steps execute later on the maintainer's machine.
   Instead, write a task file at `.agent/tasks/issue-<n>.json`:

   ```json
   {
     "issue": <n>,
     "command_args": "<arguments the worker command needs, if any>",
     "expected_artifacts": ["<paths or globs the run must produce>"]
   }
   ```

   The local worker runs the command configured in `.agent/config.json`, substituting `{task_file}` with this file's path.
5. Open a PR from the branch with `Closes #<n>` in the body.
   Summarize what was implemented and how it meets the acceptance criteria.
6. Update labels (always pass `--repo kwu93/nanoGPT` to gh):
   - If you wrote a task file: `gh pr edit <pr> --add-label needs-local-run`
   - `gh issue edit <n> --remove-label queued --add-label in-progress`

## Labels

| Label | Meaning |
| --- | --- |
| `agent-task` | Issue is a unit of agent work. |
| `queued` | Issue is waiting for an implementer. |
| `in-progress` | An agent has opened a PR for the issue. |
| `needs-local-run` | PR is waiting for the local worker to execute its task file. |
| `local-run-done` | The local worker ran the task and pushed the artifacts. |

## Reviewing a PR

When asked to review, check the acceptance criteria from the linked issue, correctness of the change, and that nothing marked local-run was wired into CI.
Post concrete, actionable comments.

## Project specifics

This is a research repo: from-scratch char-level language models on tiny Shakespeare, run as pre-registered experiments.
`PROCESS.md` is the authoritative experiment contract; read it before implementing any experiment issue.

Layout: root `model.py` / `train.py` / `sample.py` / `config/` are karpathy's original nanoGPT, kept untouched; Kevin's work is `solo.py` and `experiments/` (model registry, `train(config)`, sweep CLI, classical n-gram baselines).
The journal: `JOURNAL.md` is the index and living scoreboard; `journal/page1.md` and `journal/page2.md` are the frozen pre-restructure log.
Environment is pixi: `pixi run train`, `pixi run sweep <exp_dir>`, `pixi run sample`.

Implementing an experiment issue:

1. Copy `experiments/runs/_template/` to `experiments/runs/NNN-<slug>/` (next free id).
2. Write `spec.md` (Question / Method / falsifiable Predictions with numeric bands, riskiest call labeled) and `config.py` (`BASE` from a named protocol in `experiments/protocols.py`, plus `GRID` and `SEEDS`).
   Predictions must be committed before any result exists; that commit is the pre-registration.
3. Do NOT run the sweep in the implementing session; it is a local-run step.
   Task file: `command_args` is the experiment directory path, `expected_artifacts` is `["experiments/runs/NNN-<slug>/runs.jsonl"]`.
4. After the worker pushes results, `report.md` gets written (verdict per prediction, conclusion) and `JOURNAL.md` gets an index row plus any scoreboard change; that step may be a follow-up request on the PR.

Domain conventions: protocols are append-only (new version, never edit in place); every reported number must be traceable to rows in the experiment's `runs.jsonl`; checkpoints (`*.pt`) are never committed; scoreboard entries at `ref-v2` scale quote `best_val_loss`.
Style: in Markdown, each full sentence on its own line; use plain dashes, never an em dash; never add an agent name as commit co-author.

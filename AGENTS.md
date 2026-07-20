# Agent guide for kwu93/nanoGPT

This repo uses the agent-kit issue-queue workflow.
Issues labeled `agent-task` are implemented by agents on branches named `claude/issue-<n>`, and PRs close their issue with `Closes #<n>`.
Steps a task marks as "local run" are never executed in CI; they are described in `.agent/tasks/issue-<n>.json` and executed by a local worker on the maintainer's machine (PR labels `needs-local-run` -> `local-run-done`).
The full contract lives in `CLAUDE.md`.

## Review guidelines

When reviewing a PR (for example via `@codex review`):

- Verify the change meets the acceptance criteria in the linked issue.
- Verify no local-run step leaked into CI workflows; training, GPU work, and local data access belong in the task file, not in `.github/workflows/`.
- If the PR adds a task file, sanity-check it: valid JSON, issue number matches, `expected_artifacts` are plausible paths, and the requested run is reasonably sized.
- Flag only real problems (correctness, security, data loss); skip style nits.

## Project specifics

Research repo: from-scratch char-level language models on tiny Shakespeare, run as pre-registered experiments; `PROCESS.md` is the authoritative contract.
One experiment = one directory `experiments/runs/NNN-<slug>/` (spec.md, config.py, runs.jsonl, report.md) = one PR; sweeps execute only via the local worker.
Root `model.py` / `train.py` are karpathy's originals and stay untouched; Kevin's code is `solo.py` and `experiments/`.

Extra review checks for experiment PRs:

- The spec commit (predictions) must precede the results commit; pre-registration is the point.
- Every number in `report.md` must be traceable to rows in that experiment's `runs.jsonl`.
- `experiments/protocols.py` is append-only; flag any edit to an existing protocol.
- `*.pt` checkpoints must never be committed; `experiments/runs/_template/` stays untouched.

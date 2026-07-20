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

<!-- agent-kit: fill this in per repo. Build/test commands, code layout, domain conventions. -->

# Review rules for kwu93/nanoGPT

This repo uses the agent-kit issue-queue workflow (see `CLAUDE.md` for the full contract).

When reviewing a PR:

- Check the change against the acceptance criteria in the linked issue (`Closes #<n>` in the PR body).
- Training runs, GPU work, and local-data steps must never run in CI.
  They belong in `.agent/tasks/issue-<n>.json`, executed by the local worker.
  Flag any PR that wires such steps into `.github/workflows/`.
- If the PR adds `.agent/tasks/issue-<n>.json`: it must be valid JSON, reference the right issue, and list plausible `expected_artifacts`.
- Prioritize correctness, security, and data-loss risks over style.

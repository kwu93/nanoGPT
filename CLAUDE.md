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

<!-- agent-kit: fill this in per repo. Build/test commands, code layout, domain conventions, what a good artifact looks like. -->

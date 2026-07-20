# Operating model for experiments

This repo runs a research loop: pre-registered hypotheses, sweeps, and reports.
This file is the contract for how an experiment moves from question to merged record.
It applies to humans and agents equally.

## The unit of work

One experiment = one directory = one branch = one PR.

```
experiments/runs/NNN-<slug>/
  spec.md      # Question / Method / pre-registered Predictions; committed BEFORE any run
  config.py    # BASE (from a named protocol), GRID, SEEDS; the exact swept knobs
  runs.jsonl   # results for this experiment only (committed; one JSON line per run)
  curves/      # figures and CSVs (committed)
  report.md    # Results / verdict per prediction / Conclusion
  models/      # checkpoints (never committed; *.pt is gitignored)
```

`NNN` is the next free three-digit id; `experiments/runs/_template/` holds the skeletons.

## Lifecycle

1. **Pre-register.**
   Branch `exp/NNN-<slug>`, write `spec.md` and `config.py`, commit, open the PR.
   The first commit's timestamp is the proof that predictions preceded data.
   Predictions must be falsifiable, with numeric bands, and the riskiest call labeled.
2. **Run.**
   `pixi run sweep experiments/runs/NNN-<slug>`.
   The sweep reads `config.py`, appends rows to the experiment's own `runs.jsonl`, and stamps every row with `exp`, `git_sha`, `git_dirty`, and a UTC timestamp.
   Commit the results and any figures to the branch.
3. **Report.**
   Write `report.md`: results, an explicit verdict on every prediction, and a conclusion.
   Add one row to the experiment index in `JOURNAL.md` and update the scoreboard if a standing number changed.
4. **Merge.**
   Reports always merge, including dead ends and refuted predictions; refutations are the most informative entries.
   Merging is a record, not a quality gate; self-merge freely.

## Merge policy for code

Framework changes (new model, new config key, new metric) that an experiment needed graduate into `experiments/` core as part of the experiment's PR.
If the experiment was a dead end but the framework change is good, cherry-pick it into its own small PR.
Checkpoints never merge; `config` + `seed` in a row regenerate any model deterministically.

## Protocols

Named base configs live in `experiments/protocols.py` (`std-v1`, `ref-v2`, ...).
A spec says `base: ref-v2` instead of restating hyperparameters.
Protocols are append-only: to change the reference setting, add `ref-v3` and leave `ref-v2` frozen.

## Provenance rules

Every number in a report must be traceable to rows in that experiment's `runs.jsonl`.
Run sweeps from a clean tree at the pre-registration commit whenever possible; `git_dirty: true` in a row flags a deviation.
Fixed seeds (0, 1, 2 unless the spec says otherwise); scoreboard entries at `ref-v2` scale quote `best_val_loss`.

## Cross-experiment analysis

Aggregate with a glob over `experiments/runs/*/runs.jsonl`; each row's `exp` field says where it came from.

## Legacy layout (pre-restructure, frozen)

`journal/page1.md` and `journal/page2.md` hold the 25 entries from 2026-07-12 to 2026-07-16.
The root `runs.jsonl` (460 rows) and root `curves/` are their evidence, committed as-is and append-frozen; new sweeps cannot write there (the CLI requires an experiment directory).
`JOURNAL.md` is the living index and scoreboard.

## Planned: agent-kit

Once stamped, the flow maps 1:1: an issue is the Question, the drain agent writes `spec.md` + `config.py` and opens the PR, the local worker runs the sweep and pushes `runs.jsonl` + `curves/` back to the branch, and PR review is the science review.
Queue mechanical experiments; keep hypothesis-forming and verdicts interactive when the question matters.

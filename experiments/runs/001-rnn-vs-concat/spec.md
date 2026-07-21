# 001: rnn vs mlp_concat k=8 under ref-v2

- Status: pre-registered
- Base protocol: ref-v2
- Command: `pixi run sweep experiments/runs/001-rnn-vs-concat`
- Issue: kwu93/nanoGPT#1

## Question

At the page-1 standard setting the rnn beat `mlp_concat` at every matched-params tier: the ~0.05 gap was scale-stable from 3.9k to 60k params (learning #15), so weight sharing's win was not a scarcity artifact.
Does that survive the regularized reference protocol - 4x capacity, 16x tokens/step, weight decay 0.1, checkpoint-at-val-min - and if the rnn wins, which mechanism carries it?

Candidate mechanisms (from the issue):

- H1 weight sharing / translation invariance: one transition function serves every offset, so each pattern is learned once from all positions; concat learns per-offset features that cannot transfer.
- H2 unbounded context: the rnn state reads all 32 chars of the training window while concat k=8 is structurally blind past 8; any win could be context, not recurrence.
- H3 soft back-off off-support: contractive recurrence decays old information, so the state is dominated by the recent suffix, functionally closer to KN's back-off policy (the KN-anatomy entry located concat's deficit off-support, +0.434 at d<=3).
- H4 implicit regularization: params are independent of context length (103,041 at E64/H256 vs concat k=8's 152,193) and the shared cell must serve all positions, predicting a smaller train/val gap under the same protocol.
- H5 depth of composition: the rnn applies t nonlinear steps per prediction; concat gets one hidden layer.

This experiment separates H1 from H2 with a new `context_k` option on the rnn and prices H4 via train/val gaps.
H3 is scoped out to a follow-up (per-d-bucket scoring vs KN with the kn_anatomy machinery) if the rnn wins.
H5 is confounded with H1 in this design (any rnn cell composes depth-of-window steps); it is noted, not tested.

## Method

Framework change graduating with this experiment: the rnn gains `context_k` (windowed recurrence).
Position t's state is rebuilt from h=0 over the last min(t+1, k) tokens, so every prediction is structurally blind past k chars, matching concat's reach at the same k.
Absent or None keeps the pre-existing unbounded behavior, `context_k >= seq_len` coincides with it, and no parameters change, so existing checkpoints load unmodified.
Short prefixes (t < k-1) recurse over only the real tokens from h=0, exactly as the unbounded path treats them; concat instead sees zero-padded embedding slots there, a convention difference confined to the first k-1 of 32 positions.

Grid: 2x2 cross of `context_k` in {None, 8} and `dim_hidden` in {256, 325}, 3 seeds, 12 runs, all at seq_len 32 / wd 0.1 / lr 1e-3 / 10k iters per `ref-v2`.
H=325 is the param-matched width: 152,100 params vs concat k=8's 152,193 (the H=256 cells sit at 103,041).

Scoring: mean `best_val_loss` over seeds 0/1/2 (the ref-v2 scoreboard convention); gap = final val minus final train.

Protocol parity with the concat baseline: wd-only vs wd-only.
The rnn gets no dropout support here, so the comparator is concat k=8 at wd 0.1 alone, best 1.776 +/- .005 (train/val 1.506/1.781, gap 0.275; page 2, weight-decay entry), with the ladder optimum k=5 at 1.772 as a secondary reference.
The wd+dropout number (1.769, page 2, dropout entry) is NOT a comparator: mismatched regularizers.

Readings:

- H1 at matched context: rnn k=8 vs concat k=8, at H=256 (rnn param-disadvantaged) and at H=325 (params matched too).
- H2 price: rnn unbounded vs rnn k=8 within each width tier.
- H4: every cell's train/val gap vs concat k=8's 0.275.

Deliverables: report.md with a cell table (best +/- seed spread, @iters, gap), a verdict per prediction, the H1/H2/H4 readings, a train/val curves figure in curves/, a JOURNAL.md index row, and a scoreboard entry for the rnn at ref-v2 scale.

Local run: CPU training on the maintainer's machine via the task file; the rnn's python-loop recurrence is slower per step than the MLPs (the windowed cells loop k=8 times over B*T-row matmuls, the unbounded cells 32 times over B-row matmuls); actual wall time goes in report.md.

## Predictions (pre-registered)

Committed and pushed before any run executes; the commit timestamp is the pre-registration proof.
Predictions were written by the implementing agent from the journal record, per the issue-queue workflow.

1. Levels: rnn unbounded H=256 lands at mean best_val in [1.68, 1.77], above KN's 1.634 (extrapolating from 1.827 at 31k params / 60k iters / no wd on ~10M tokens to 3.3x params, ~20M tokens, wd 0.1, checkpointing).
2. Headline: rnn unbounded H=256 beats concat k=8 wd-only by at least 0.01 - mean best_val <= 1.766.
3. H1 dominant: rnn k=8 H=256 also beats 1.776, and retains at least half of the unbounded cell's margin: (1.776 - best(k8, H256)) >= 0.5 x (1.776 - best(unb, H256)) > 0.
4. H2 price: best(k8, H256) - best(unb, H256) lands in [0.01, 0.05] (chars 9-32 are worth something, but the KN ladder says most value is in the recent suffix).
5. H4: every rnn cell's final train/val gap is below concat k=8's 0.275, with the unbounded H=256 gap in [0.08, 0.22].
6. Width secondary: each H=325 cell improves on its H=256 counterpart by [0.00, 0.03] (the params frontier showed diminishing returns well before this scale).
7. Riskiest call: at matched params AND matched context - rnn k=8 H=325 (152,100) vs concat k=8 (152,193, 1.776) - the page-1 gap survives regularization: the rnn wins by [0.02, 0.08], i.e. best_val in [1.696, 1.756]. Risky because wd 0.1 specifically repaired concat's memorization deficit; if the rnn's page-1 edge was mostly implicit regularization (H4), explicit regularization buys concat the same thing and this gap collapses toward 0.

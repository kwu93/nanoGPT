# Research Journal, page 2

Continuation of `page1.md`, same conventions: log the proposed explanation as a hypothesis with a falsifiable prediction *before* running the experiment, then record the result.
Page 1 closed with the scoreboard at the old standard setting (dim_embed 32, dim_hidden 128, 5000 iters); this page opens the larger reference scale.

## 2026-07-16: Dropout 0.2 on top of weight decay - the concat ladder

**Question.**
The weight-decay entry (below) ended with headroom on the table: wd 0.1 shrank but did not close the memorization gap (0.24-0.38 across the concat ladder, train at k=32 still 1.476), and flagged dropout as the orthogonal regularizer to try next (bigram.py uses 0.2).
Does stacking dropout 0.2 on wd 0.1 keep converting overfitting into val gains, and does the k=5/k=8 tie finally break toward deeper context?

**Method.**
Models gain a `dropout` config key: the shared two-layer MLP block gets `nn.Dropout` on the hidden activations (after the ReLU), active only in train mode; eval-mode losses stay noise-free and comparable to earlier entries.
Placement note: bigram.py drops the FFN output and the attention weights, but this block's hidden layer is the only interior surface, so the ReLU output is the analogous site.
Ladder: mlp_concat only (the plot under construction is the concat figure; the bag ladder waits), context_k in {1, 2, 3, 4, 5, 8, 16, 32}, 3 seeds, seq_len 32, E64/H256, batch 64, lr 1e-3, wd 0.1, dropout 0.2, num_iters 10,000, eval_every 1,000, checkpoint-at-val-min - the wd entry's protocol with one new key.
24 runs appended to runs.jsonl, best-val weights in models/, 7 CPU workers.
Deliverable: `curves/mlp_concat_train_val_vs_context_wd_dropout.png`, same anatomy as the no-regularization figure `mlp_concat_train_val_vs_context_e64h256.png`.

**Prediction (pre-registered, by the agent).**
1. Dropout costs train fit everywhere: final train NLL rises at every k versus the wd-only ladder (k=32 from 1.476 to >= 1.55), and the train/val gap shrinks at every k >= 3, with the k=32 gap below 0.30 (from 0.375).
2. Shallow cells pay for noise they did not need: k=1 best-val worsens to [2.500, 2.540] (wd-only: 2.498), and k=2 lands within +/- 0.015 of its wd-only 2.082.
3. Mid and deep cells gain: k=8 best-val improves to [1.730, 1.770] (from 1.776), and k=16 and k=32 each improve by at least 0.02 (from 1.818 and 1.843).
4. Riskiest call: the k=5/k=8 tie breaks - k=8 becomes the optimum with non-overlapping seed ranges against k=5, the first time this ladder's optimum moves past 5.
5. Convergence slows further: at least half the cells with k >= 4 have best_iter pinned at 10,000 in 2 or more seeds.
6. KN (1.634) survives its fifth challenger: best cell >= 1.70.

**Result.**
Same reporting as the wd entry: best = mean best_val_loss over 3 seeds, @iters = per-seed best_iter, gap = final val minus final train; "wd best" = the wd-only entry's checkpoint column, "vs" = this ladder's best minus that.
24 runs appended to runs.jsonl (~25 min wall on 7 CPU workers); checkpoints in models/; figure in `curves/mlp_concat_train_val_vs_context_wd_dropout.png`.

| k | train / val | best +/- | @iters | gap (wd: was) | wd best | vs |
|---|---|---|---|---|---|---|
| 1 | 2.453 / 2.501 | 2.497 +/- .002 | 9,9,8k | 0.048 (0.047) | 2.498 | -0.001 |
| 2 | 1.949 / 2.090 | 2.085 +/- .002 | 6,6,8k | 0.142 (0.140) | 2.082 | +0.003 |
| 3 | 1.674 / 1.869 | 1.864 +/- .004 | 9,10,8k | 0.195 (0.197) | 1.852 | +0.012 |
| 4 | 1.584 / 1.798 | 1.793 +/- .004 | 9,9,8k | 0.214 (0.227) | 1.789 | +0.004 |
| 5 | 1.555 / 1.781 | 1.776 +/- .005 | 9,9,8k | 0.226 (0.240) | 1.772 | +0.004 |
| 8 | 1.530 / 1.775 | **1.769** +/- .005 | 9,10,8k | 0.245 (0.275) | 1.776 | -0.007 |
| 16 | 1.514 / 1.808 | 1.804 +/- .008 | 9,9,8k | 0.294 (0.341) | 1.818 | -0.014 |
| 32 | 1.513 / 1.819 | 1.815 +/- .007 | 9,10,10k | 0.307 (0.375) | 1.843 | -0.028 |

Prediction 1 direction right, magnitudes overestimated: the gap shrinks at every k >= 3 (confirmed), but train rose far less than called (k=32 lands at 1.513, not >= 1.55; k=1 did not rise at all, 2.453 vs 2.454), and the k=32 gap misses < 0.30 by a hair (0.307).
Prediction 2 half-refuted: k=1 did not pay for the noise (best 2.497, a hair *better* than wd-only's 2.498 - at 20% drop a 256-unit hidden layer is redundant enough to shrug), while k=2 landed +0.003 from its wd-only value, in-band.
Prediction 3 mixed: k=8 confirmed at 1.769, just inside [1.730, 1.770]; k=32 confirmed (+0.028 improvement); k=16 missed its >= 0.02 clause (improved 0.014).
Prediction 4 half-confirmed, and the half that held is the ladder's first: the optimum moved past k=5 for the first time under any protocol - k=8 leads under both estimators (best 1.769 vs 1.776, final val 1.775 vs 1.781) - but the non-overlap clause fails (k=8 seeds span [1.763, 1.774], k=5 [1.770, 1.783], touching in [1.770, 1.774]).
And the move is half push, half pull: k=8 fell 0.007 while k=3-5 *rose* 0.004-0.012 - dropout taxes the cells that were not overfitting and pays where they were.
Prediction 5 REFUTED: only k=32 pins best_iter at 10k in 2+ seeds; every other cell's minima sit at 8-9k, exactly where wd-only put them - dropout changed the level, not the schedule.
Prediction 6 confirmed: KN (1.634) survives its fifth challenger; the margin ticks down 0.138 -> 0.135.
Drift stays a rounding error (+0.004 to +0.006 everywhere).

**Conclusion / next steps.**
Dropout on top of weight decay is a tail treatment, not a floor treatment: the deep cells gain in proportion to the gap they had left (-0.007 at k=8, -0.014 at k=16, -0.028 at k=32), the shallow-mid cells pay a small noise tax (+0.004 to +0.012), and the ladder minimum moves only 0.003 (1.772 -> 1.769).
The U keeps flattening on schedule: the k=32 penalty over the optimum is now 0.046, down from 0.087 (early-stopped, no reg) and 0.071 (wd-only), and the optimum sits at k=8 for the first time - regularization keeps dragging the optimum right, but by ever-smaller margins.
The regularization direction is close to exhausted: two stacked regularizers cut KN's margin from 0.138 to 0.135, and the k=32 memorization gap is still 0.31 (train 1.513) - consistent with the KN-anatomy entry's diagnosis that the binding constraint is representational (fixed-window weights with no off-support policy), not statistical.
Standing next steps unchanged: the lambda sweep at k=5/8, the random-context-truncation causal test, and the rnn / lstm / attention scale-up under the regularized protocol.

## 2026-07-16: Anatomy of the KN gap - coverage, counts, and effective depth

**Question.**
With scoring matched (full-context val positions only), regularized concat k=8 sits at ~1.738 against KN's 1.659 at the same depth and 1.634 at its optimum.
The proposed mechanism, so far argument rather than measurement: KN wins exactly where exact counts exist, backs off near-optimally where they do not, and never pays for depth it does not use, while the MLP compresses the same table through a low-rank bottleneck (losing the covered contexts) but generalizes (competitive off-support).
Four tables to turn that into evidence: (A) how much of val is covered by train contexts at each depth, (B) the KN-vs-concat gap conditioned on the longest matched suffix depth, (C) the gap conditioned on the training count of the full context, (D) the depth at which KN's recursion actually grounds when offered 32 characters.

**Method.**
Shared prediction set: every val position i >= 32 (the classical ladder's protocol), ~223k predictions.
concat: mean per-position NLL over the three wd-0.1 checkpoints (k=8 cell), each scored from its exact 8-char context; KN: the depth-8 model built on train, per-position prob(window, next, 8).
Longest matched suffix depth d per position = the largest d <= 32 such that the d-char context appears anywhere in train (suffix matches are nested, so coverage at depth k is P(d >= k)); context counts from train n-gram tables.
Table A rows are the ladder depths {1, 2, 3, 4, 5, 8, 16, 32}: coverage, mean and median matched count, singleton share of matches.
Table B buckets positions by d (8, 7, 6, 5, 4, <= 3) and reports each model's NLL and the gap.
Table C buckets by the train count c of the full 8-char context (0, 1, 2-4, 5-19, 20+).
Table D is the distribution of d with no cap short of 32: mean and the P(d >= j) milestones.
Sanity ties: aggregate KN on this set must reproduce ~1.659, aggregate concat ~1.74.

**Prediction (pre-registered, by the agent).**
1. Coverage falls off a cliff after k=5: k=4 in [90, 98]%, k=5 in [80, 93]%, k=8 in [35, 55]%, k=16 in [1, 8]%, k=32 < 0.5%; among k=8 matches the median count is 1 or 2 and singletons are 40-60%.
2. KN's edge is monotone in d: it wins the d=8 bucket by 0.10-0.30 nats, the gap shrinks as d falls, and (riskiest clause) concat *wins* at least one bucket with d <= 4: generalization beats back-off once counting is off-support.
3. The counts lens shows an inverted U: KN's edge peaks at moderate counts (c in 2-19, evidence solid for a table, too sparse for SGD to carve sharp rules), is smaller at c >= 20 (frequent patterns are easy for both), and is smallest or negative at c = 0.
4. Deep context is free because it is unused: at nominal k=32 the mean effective depth is in [6.5, 8.5], P(d >= 12) <= 10%, P(d >= 16) <= 6%.

**Result.**
Sanity ties held: KN aggregates to 1.6589 on the scored set (ladder: 1.659), concat to 1.7394 (masked check: 1.738); total gap 0.0805.

A. Coverage of the ~223k scored val positions by train contexts: the share whose exact k-char context appears anywhere in train.

| k | coverage |
|---|---|
| 1 | 100.0% |
| 2 | 99.8% |
| 3 | 97.7% |
| 4 | 93.9% |
| 5 | 87.3% |
| 8 | 47.2% |
| 16 | 1.6% |
| 32 | 0.0% |

The count spectrum behind coverage (median train occurrences of the matched context: 3,494 at k=2, 30 at k=5, 4 at k=8, 1 at k=16) matters in one place only: KN's depth-8 level typically rests on ~4 observations that discounting mostly dissolves back into shallower levels, which is why depth 8 buys KN nothing over depth 5.
Table C below shows the concat-KN gap is insensitive to count size once coverage is nonzero, so evidence *quantity* is not the operative variable for the comparison - coverage and matched depth are.

B. NLL by longest matched suffix depth d (gap = concat minus KN):

| bucket | share | KN | concat | gap |
|---|---|---|---|---|
| d=8 | 47.2% | 1.561 | 1.591 | +0.030 |
| d=7 | 14.8% | 1.564 | 1.622 | +0.058 |
| d=6 | 14.1% | 1.592 | 1.667 | +0.075 |
| d=5 | 11.2% | 1.680 | 1.787 | +0.107 |
| d=4 | 6.6% | 1.906 | 2.037 | +0.131 |
| d<=3 | 6.1% | 2.487 | 2.921 | **+0.434** |

C. NLL by train count c of the full 8-char context:

| bucket | share | KN | concat | gap |
|---|---|---|---|---|
| c=0 | 52.8% | 1.746 | 1.872 | +0.126 |
| c=1 | 12.2% | 1.593 | 1.617 | +0.024 |
| c=2-4 | 13.6% | 1.553 | 1.583 | +0.029 |
| c=5-19 | 13.5% | 1.573 | 1.610 | +0.037 |
| c>=20 | 7.8% | 1.504 | 1.533 | +0.028 |

D. Effective grounding depth at nominal k=32: mean d = 7.60; P(d >= 8) = 47.2%, P(d >= 12) = 9.3%, P(d >= 16) = 1.6%, P(d >= 20) = 0.3%.

Prediction 1 half-confirmed: every coverage band hit (93.9 / 87.3 / 47.2 / 1.6 / 0.0), but the matched-evidence texture was underestimated: the median k=8 match has count 4 and only 25.9% of matches are singletons.
Prediction 2 REFUTED in both clauses, and this is the entry's finding: the gap at d=8 is +0.030, *below* the predicted 0.10-0.30, and it *grows* monotonically as d falls (+0.030 -> +0.434) instead of shrinking; concat never wins a single bucket.
Prediction 3 REFUTED: no inverted U - the gap is flat and small (+0.024 to +0.037) across every covered count bucket including c >= 20, and the aggregate gap concentrates entirely at c=0 (+0.126 over 52.8% of positions).
Prediction 4 confirmed: mean effective depth 7.60 with P(d >= 16) = 1.6% - KN offered 32 characters grounds its predictions at depth ~4-10 essentially always, which is mechanically why its deep tail is free.
Decomposition (bucket share x gap, reconstructing the 0.0805 total): the d <= 5 positions - 24% of val - contribute 59% of the gap, and the d <= 3 positions - 6% of val - contribute 33% alone.

**Conclusion / next steps.**
The compression story is dead: on covered contexts the 152k-float MLP sits 0.024-0.037 nats from exact counting at *every* evidence level, including the singleton bucket - squeezing the seen part of the table through a 64-dim bottleneck is nearly lossless, and "sharp exceptions cost the smooth model capacity" is measured to be a red herring at this scale.
KN's real edge is an off-support *policy*, not a memory: when the deep context is novel, back-off discards the unmatched prefix and grounds in continuation counts at whatever depth still has evidence, while concat must consume all 8 characters through fixed weights with no mechanism to detect that part of its input is off-manifold - and the penalty for that escalates from +0.126 (novel 8-context) to +0.434 (novel 4-context).
This inverts the standing complementarity story: it is not "KN memorizes, the net generalizes" but "both estimate seen contexts almost equally well; KN *generalizes better*" - Pitman-Yor-style back-off is a stronger off-support generalizer than learned smoothness in a one-layer window model.
It also gives the architecture ladder a sharper reading: adaptive effective context - attend to the suffix that has evidence, ignore the prefix that does not - is exactly what dynamic selection (attention) can represent and fixed per-offset weights cannot, which reframes why attention wins at scale despite losing every tiny-scale comparison.
Next, the causal test: train concat k=8 with random context truncation (mask a random-length prefix each step, teaching the model to predict from short suffixes) - if the d <= 3 bucket gap closes materially, the off-support diagnosis is confirmed; score the existing attention checkpoints per-d-bucket for the same signature; and refresh the KN mixture with per-bucket weights now that we know *where* each model wins.

## 2026-07-16: Weight decay + checkpoint-at-val-min - does regularization convert drift into gains?

**Question.**
The E64/H256 ladder entry (below) found every concat cell with k >= 3 over-trained: val minima at ~3-8k iters, then drift of +0.03 to +0.11 nats by 20k, gaps up to 0.62, and a deep tail (k=16/32) that loses to the smaller page-1 models at final val purely through overfitting.
Two protocol fixes go in now: train() checkpoints at the best val eval and returns/saves those weights (with best_val_loss / best_iter recorded per run), and the optimizer becomes AdamW with decoupled weight decay 0.1 (nanoGPT's default; applied to *all* parameters including embeddings and biases, a simplicity deviation from nanoGPT's param-group split).
Does weight decay close the gaps, push the val minima later, and let the deep-context cells finally cash in the width they paid for - and does the optimum context finally move past k=5?

**Method.**
Same grid as the previous entry: {mlp_concat, mlp_sum} x k in {1, 2, 3, 4, 5, 8, 16, 32}, 3 seeds, seq_len 32, E64/H256, batch 64, lr 1e-3, appended to runs.jsonl, best-val weights in models/.
New protocol per the updated BASE_CONFIG (post-budget-finding): num_iters 10,000 (~23 epochs, covering all observed no-wd minima), eval_every 1,000 (11 curve points; checkpoint granularity = 1k iters).
Scoring: best_val_loss (the checkpoint) is the headline number now; final-iterate val is kept for drift measurement.
Cross-entry caveat: the no-wd baseline ran 20k iters, so final-val comparisons are budget-confounded; checkpoint-to-checkpoint comparisons are the honest ones (both protocols bracket their minima unless best_iter pins at 10k).

**Prediction (pre-registered, by the agent).**
1. Mechanics: best_val_loss <= final val for every run, and the k=1 twins remain bit-identical per seed under the new optimizer (decay consumes no RNG).
2. Gaps shrink: concat k=32 final gap drops below 0.40 (from 0.619 at 20k), and every concat cell k >= 3 has a smaller gap than its no-wd counterpart had at iter 10k.
3. Drift becomes negligible: final minus best <= 0.03 for every concat cell (no-wd reached +0.11).
4. Levels and the riskiest call: the concat best-val ladder beats the no-wd early-stopped ladder at every k >= 8 (regularization is worth most where params/data is worst), the optimum moves to k=8 with best-val in [1.75, 1.81], and the tail flattens (k=32 best-val within 0.06 of the optimum).
5. Budget check: weight decay delays convergence, so at least two concat cells with k >= 4 end with best_iter = 10,000 (val still falling at the budget boundary), reopening the budget question at this lambda.
6. mlp_sum barely moves: optimum stays k=3 with best-val in [2.26, 2.31], shallow cells (k=2-4) gain 0.00-0.04 from the gap reduction, deep cells (k >= 8) change within +/-0.05 (representation-bound; decay cannot buy back dilution).
7. KN (1.634) survives its fourth challenger: best cell >= 1.70.

**Result.**
best = mean best_val_loss over 3 seeds (the checkpoint, the headline number); @iters = per-seed best_iter; gap = final val minus final train; "no-wd es" = the previous entry's early-stopped ladder at 20k iters.
48 runs appended to runs.jsonl (~35 min wall on 7 CPU workers; the halved budget and 8x-coarser eval cadence cut the previous entry's ~2h to a third); checkpoints in models/; figures in curves/*_wd.png.

| k | concat train / val | best +/- | @iters | gap | no-wd es |
|---|---|---|---|---|---|
| 1 | 2.454 / 2.501 | 2.498 +/- .002 | 9,9,8k | 0.047 | 2.499 |
| 2 | 1.947 / 2.087 | 2.082 +/- .002 | 9,8,8k | 0.140 | 2.094 |
| 3 | 1.659 / 1.856 | 1.852 +/- .002 | 9,9,8k | 0.197 | 1.880 |
| 4 | 1.565 / 1.793 | 1.789 +/- .003 | 9,10,10k | 0.227 | 1.823 |
| 5 | 1.535 / 1.775 | **1.772** +/- .002 | 9,10,10k | 0.240 | 1.811 |
| 8 | 1.506 / 1.781 | 1.776 +/- .005 | 9,10,8k | 0.275 | 1.827 |
| 16 | 1.483 / 1.823 | 1.818 +/- .005 | 9,10,8k | 0.341 | 1.873 |
| 32 | 1.476 / 1.851 | 1.843 +/- .008 | 9,8,6k | 0.375 | 1.898 |

| k | sum train / val | best +/- | @iters | gap | no-wd es |
|---|---|---|---|---|---|
| 1 | 2.454 / 2.501 | 2.498 +/- .002 | 9,9,8k | 0.047 | 2.499 |
| 2 | 2.169 / 2.292 | 2.290 +/- .002 | 6,10,10k | 0.123 | 2.303 |
| 3 | 2.084 / 2.272 | **2.269** +/- .003 | 9,10,10k | 0.188 | 2.291 |
| 4 | 2.175 / 2.356 | 2.351 +/- .001 | 9,10,10k | 0.180 | 2.359 |
| 5 | 2.313 / 2.493 | 2.490 +/- .004 | 9,10,10k | 0.180 | 2.500 |
| 8 | 2.517 / 2.690 | 2.685 +/- .005 | 9,10,10k | 0.173 | 2.705 |
| 16 | 2.688 / 2.835 | 2.829 +/- .005 | 9,10,10k | 0.148 | 2.842 |
| 32 | 2.751 / 2.883 | 2.878 +/- .002 | 9,10,10k | 0.133 | 2.883 |

Prediction 1 confirmed: best <= final for all 48 runs and the k=1 twins are again bit-identical per seed.
Prediction 2 confirmed in both clauses: the k=32 concat gap is 0.375 < 0.40 (from 0.619 at the 20k no-wd budget), and at matched iteration the wd gap is smaller at every k >= 3 (e.g. k=8: 0.275 vs 0.374 no-wd@10k; k=32: 0.375 vs 0.519).
Prediction 3 confirmed and exceeded: worst drift is +0.008 (checkpointing plus flatter curves make final-iterate reporting a rounding error).
Prediction 4: the first clause is confirmed everywhere it applies and then some (weight decay beats the no-wd early-stopped ladder at *every* k >= 2, with gains growing in k: -0.012 at k=2, -0.051 at k=8, -0.055 at k=32); the riskiest call is refuted in the interesting way: the optimum did not flip to k=8, it became a statistical *tie* (k=5 at 1.772 +/- .002 vs k=8 at 1.776 +/- .005, overlapping seed ranges), and the level landed in-band; the tail clause narrowly missed (k=32 sits 0.071 above the optimum, predicted <= 0.06, was 0.087).
Prediction 5 confirmed: concat k=4 and k=5 each checkpoint at iter 10,000 in 2 of 3 seeds (and nearly every sum cell k >= 2 does), so the minima now sit at or beyond the budget boundary, though the 8-10k curve segments are close to flat (drift <= 0.008 says little was left on the table at this lambda).
Prediction 6 confirmed in full: the sum optimum stays at k=3 (2.269, in-band), shallow cells gain 0.008-0.022, deep cells move 0.005-0.020, all improvements, none beyond +/-0.05: decay trims the bag's estimation noise but cannot buy back dilution.
Prediction 7 confirmed: KN survives at 1.634, but its margin over the best neural window model shrinks to 0.138 (1.901 page-1 -> 1.811 early-stopped -> 1.772 regularized).

Samples from the best checkpoints (400 chars, `\n\nISABEL` primer, seed 0) keep their family textures: concat k=5 wd (exact NLL 1.7787) writes words and speaker turns ("PRINCE:\nMy father.", "LUCIO:\nI bad lover,"), sum k=3 wd (2.2754) still writes anagram soup with intact speaker formatting ("FRIA MRS:", "OYRCATLE:").

**Conclusion / next steps.**
Weight decay converts drift into gains, full stop: at *half* the training budget every cell of both ladders beats the unregularized early-stopped ladder, and the gains grow exactly where overfitting was worst (concat -0.012 at k=2 up to -0.055 at k=32).
The deep tail is reclaiming its width: the regularized U is visibly flatter (k=32 penalty 0.071 vs 0.087 early-stopped and 0.168 at final val), and the optimum is now a k=5/k=8 tie where every previous protocol had k=5 clearly ahead: regularization is dragging the optimum right, one lambda at a time.
But wd 0.1 shrinks rather than closes the memorization gap (0.24-0.38 across the concat ladder, train at k=32 still 1.476), so there is headroom in the regularization direction before the representation binds.
Checkpoint-at-val-min retires the final-iterate convention: drift is now <= 0.008, the saved models are the ones worth sampling, and scoreboard entries at this scale should quote best_val_loss.
The classical floor is in sight: KN's margin is down to 0.138 nats.
Next: a lambda sweep ({0.03, 0.3} around 0.1) at k=5/8 to find where regularization stops paying; a 20k-iter wd run for the boundary-pinned cells (expected to buy <= 0.01); dropout as the orthogonal regularizer (bigram.py uses 0.2); and the deferred rnn / lstm / attention scale-up, now under the wd + checkpoint protocol.

## 2026-07-16: The context ladders at 4x capacity and 8x tokens (E64/H256, 20k iters)

**Question.**
The page-1 context ladders ran mlp_concat and mlp_sum at E32/H128 with 5.12M training tokens, and two things were left hanging: the concat ladder's U-turn was partly an artifact of starving deep cells of width (the 141.6k capacity-matched cell moved the optimum from k=5 to k=8), and every curve was suspected under-trained (seed-0 curves still falling 0.07-0.10 nats over the final 2000 iters).
The new BASE_CONFIG quadruples capacity (dim_embed 64, dim_hidden 256) and raises the budget to 20,000 iters at batch 64 x seq 32 = 2,048 tokens/step (41M tokens, ~46 epochs of the 892k-char train split).
Where do the two ladders land when both width and training budget stop binding, is 20k iters actually enough to call the runs converged, and what does the best model of each family write?

**Method.**
`sweep` over model x context_k with BASE_CONFIG + seq_len 32: model in {mlp_concat, mlp_sum}, context_k in {1, 2, 3, 4, 5, 8, 16, 32}, 3 seeds, appended to runs.jsonl, weights in models/.
Everything else from BASE_CONFIG: dim_embed 64, dim_hidden 256, batch 64, lr 1e-3, num_iters 20,000, eval_every 250 (81 curve points per run, eval_iters 100).
Params: mlp_sum constant at 37,505 for every k; mlp_concat is 21,121 + 16,384k (37,505 at k=1 up to 545,409 at k=32), so as before the concat k axis deliberately confounds context with params, because per-offset weights are the mechanism.
Loss is position-averaged over all 32 positions (warm-up caveat, page-1 learning #9), and supervision is 2x the old ladder's tokens/step (learning #10 caveat on cross-page comparisons).
Convergence is judged from the recorded curves: per cell, the val slope over the final 2,000 iters, and the gap between final val and the curve's minimum.
Sampling: 400 chars from the shared `\n\nISABEL` primer (seed 0, the sample.py convention) from the best seed-0 model of each family.
Page-1 anchors for reference: H=128 ladder minima were sum 2.323 at k=2 and concat 1.901 at k=5; the 141.6k concat cell at k=8 reached 1.833; KN stands at 1.634.

**Prediction (pre-registered, by the agent).**
1. k=1 twins: concat and sum at k=1 are identical architectures, so per-seed losses agree to float precision, landing in [2.49, 2.51] (the neural bigram floor does not move with capacity or budget, learning #1).
2. mlp_sum keeps its U anatomy unchanged: minimum at k=2 in [2.26, 2.33] (capacity and tokens buy only a little where representation binds), crossover to worse-than-bigram at k=4 or 5, k=32 in [2.75, 2.93], and the deep tail's train column *rises* with k while the gap stays under 0.10 even at 46 epochs; representation failure, not overfitting.
3. mlp_concat's optimum moves deeper with the width squeeze removed: minimum at k=8 or 16 with val in [1.72, 1.83] (below the 141.6k cell's 1.833, thanks to 8x tokens), and the U-turn is milder than the H=128 ladder's: k=32 final val within 0.10 of the ladder minimum.
4. Overfitting finally arrives where it should: the concat train/val gap grows monotonically in k and exceeds 0.30 at k=32 (545k params x 46 epochs), and the k=16 and k=32 concat cells pass their val *minimum* before iter 20,000 and end above it.
5. On the under-training question: every cell is flat or rising over the final 2,000 iters (|slope| < 0.01 nats), so 20k iters is *sufficient* at this scale; and the old budget was indeed short: at iter 5,000 the concat k=8 cell sits >= 0.05 nats above its eventual best.
6. The order gap at k=2 (sum minus concat) lands in [0.15, 0.25]: the fourth independent measurement of the ~0.2-nat order premium.
7. Samples: best concat (~1.7-1.8 NLL) writes mostly real words with fully formed speaker-turn formatting and no white-noise collapse (a parametric model has no uniform escape hatch); best sum (k=2, ~2.3) reads like slightly-better bigram soup; neither is grammatical past a short phrase.

**Result.**
Val is the final-iterate loss, mean +/- std over 3 seeds; 48 runs appended to runs.jsonl (7 parallel CPU workers, ~2h wall), weights in models/; figures in curves/*_e64h256.png.

| k | concat params | concat train / val | sum train / val (37,505 params) |
|---|---|---|---|
| 1 | 37,505 | 2.452 / 2.508 +/- .002 | bit-identical to concat |
| 2 | 53,889 | 1.934 / 2.118 +/- .005 | 2.154 / 2.320 +/- .005 |
| 3 | 70,273 | 1.619 / 1.905 +/- .012 | 2.039 / **2.304** +/- .004 |
| 4 | 86,657 | 1.513 / 1.862 +/- .003 | 2.122 / 2.369 +/- .003 |
| 5 | 103,041 | 1.480 / **1.840** +/- .011 | 2.262 / 2.511 +/- .007 |
| 8 | 152,193 | 1.441 / 1.883 +/- .020 | 2.476 / 2.717 +/- .002 |
| 16 | 283,265 | 1.404 / 1.966 +/- .012 | 2.652 / 2.869 +/- .006 |
| 32 | 545,409 | 1.389 / 2.008 +/- .016 | 2.716 / 2.901 +/- .004 |

Concat convergence (val at iter 5k; early-stop = mean over seeds of each run's val-curve minimum and its iteration; drift = final minus early-stop; gap = final val minus train):

| k | val@5k | early-stop @ ~iter | drift | gap |
|---|---|---|---|---|
| 1 | 2.506 | 2.499 @ 3.8k | +0.009 | 0.056 |
| 2 | 2.112 | 2.094 @ 1.9k | +0.024 | 0.184 |
| 3 | 1.894 | 1.880 @ 6.6k | +0.025 | 0.286 |
| 4 | 1.839 | 1.823 @ 7.9k | +0.039 | 0.349 |
| 5 | 1.827 | **1.811** @ 8.3k | +0.029 | 0.360 |
| 8 | 1.840 | 1.827 @ 6.7k | +0.056 | 0.443 |
| 16 | 1.891 | 1.873 @ 3.3k | +0.094 | 0.562 |
| 32 | 1.921 | 1.898 @ 3.0k | +0.110 | 0.619 |

Sum minima sit at iters 9-13k with drift <= 0.012 everywhere: the bag converges later and barely drifts.
Cross-page check at final val: the new concat ladder beats the page-1 H=128 ladder at every k <= 8 but *loses* to it at k=16 (1.966 vs 1.932) and k=32 (2.008 vs 1.954); early-stopped it beats page 1 everywhere.

Prediction 1 confirmed in its strict form: the k=1 twins are bit-identical per seed (2.5119 / 2.5062 / 2.5071), mean 2.508 inside [2.49, 2.51].
Prediction 2 half-refuted, and the refuted half is a finding: the sum minimum *moved* to k=3 (2.304 +/- .004 vs k=2's 2.320 +/- .005, separated seed ranges), the first time capacity has shifted the bag's optimum; the level clause held (2.304 inside [2.26, 2.33]), the crossover held at k=5 (2.511 vs bigram 2.508, now a tie rather than clearly worse, with k=4 at 2.369 still clearly *better* than no context), and k=32 landed in-band (2.901).
The gap clause is refuted: sum gaps are 0.17-0.27 at every k >= 2, not < 0.10, while train still rises from k=3 on (2.039 -> 2.716); at 46 epochs the bag both fails to represent *and* overfits, so the page-1 "pure representation failure" anatomy is now mixed.
Prediction 3 refuted on location and marginal on level: the concat optimum did not move (k=5 again, 1.840, with k=4 and k=8 both separated by seed ranges), 0.010 above the predicted band; the mild-tail clause fails at final val (k=32 sits 0.168 above the minimum) but holds early-stopped (1.898 - 1.811 = 0.087 < 0.10): the final-val tail tax is overfitting drift stacked on the page-1 variance tax.
Prediction 4 confirmed and exceeded: the gap is monotone in k and reaches 0.619 at k=32; and not just k=16/32 but *every* concat cell with k >= 3 passes its val minimum by iter ~3-8k and ends above it.
Prediction 5 split, and the refuted half inverts the working hypothesis: every curve is flat to +/- 0.009 over the final 2k iters (sufficiency confirmed), but concat k=8 at iter 5k sits 0.013 above its eventual best, not >= 0.05: at this scale the models are not under-trained at 20k, they are *over-trained* from ~6-8k on.
Prediction 6 confirmed: the order gap at k=2 is 2.320 - 2.118 = 0.202, the fourth independent ~0.2-nat measurement (0.200, 0.205, 0.208, 0.202).
Prediction 7 confirmed for concat, refuted for sum in an instructive direction (samples below): concat writes real and blended words with fully formed speaker turns and never derails; the sum sample is not bigram soup but *anagram* soup: word lengths, whitespace rhythm, and even the speaker format survive while letters scramble within words: order-blindness made directly visible in the generation regime.

Samples (400 chars, shared `\n\nISABEL` primer, seed 0; best cell of each family):

mlp_concat k=5 (103k params), exact val NLL 1.8416:

```text
LA:
And this when?

WARWICK:
Than you.
I tostern
And yet givent me rests?

Shepherdels!
These is
stants, your way, and cozen:
Ay, take nemity the even house of mine of Edward the Lancaster,
I carracle
To prayer in mine moon, he say, instouch'd be me in
received
to she.

King Edward's no soft the marry, whiled when he loving hath,
But noses mine to or of he
you wile shu stoout an will warlike, like
```

mlp_sum k=3 (37.5k params), exact val NLL 2.3049:

```text
LA:
lWhke itsr selt ilo'dt heark'd,
Now helsve gMyb  umse hdintgheer? hatn hoicseein  dse. I havest.

For porpiecsee alcesri;
And ea ndn mya tthe ewepe ot  otai'nd isez thi sche connts uwell dsn.
Shou mye xcppian
Eveny m oen, twihc einte.

YORK:
iNunigle in me upon,
As toin,
Spaarnnot  tsr etted maazy. Farn of hese h omote,
I'tl, mDubzzaes muor tha r oI  blosie
sTo sfu ing uindne olrd, alndy, lpig
```

**Conclusion / next steps.**
The under-training suspicion was right about page 1 and exactly backwards here: at 2,048 tokens/step and 4x width, concat val bottoms at ~6-8k iters (12-16M tokens) and the remaining 12-14k iters buy only overfitting drift, up to +0.11 nats at k=32.
So 20,000 iterations is sufficient, in fact past-sufficient: the protocol this scale needs is checkpoint-at-val-min (or regularization to make the full budget pay), not more iterations.
Final-iterate scoreboard numbers now *understate* big cells: the deep tail (k=16/32) loses to the page-1 ladder at final val purely through drift, and reclaims it early-stopped.
The optimum-k(params) relation from page 1 (k=3 at 14.7k, k=5 at the H=128 diagonal, k=8 capacity-matched at 141.6k) does not extrapolate along this ladder: at E64/H256 the optimum stays at k=5 under both estimators, because data, not width, now binds the deep cells.
The bag's ceiling is real but not perfectly rigid: 4x capacity moved the sum optimum from 2 to 3 chars and the worse-than-no-context crossover from k=4-5 out to k=5, yet the best bag value improved only 0.019 nats (2.323 -> 2.304).
Order is worth ~0.2 nats in a 2-char window for the fourth time, and generation gives the premium a face: the ordered window writes words, the bag writes anagrams of words with intact word-length and whitespace prosody.
KN (1.634) still stands 0.18 below the best cell here (1.811 early-stopped).
Next: rerun the concat ladder with dropout / weight decay to test whether regularization converts drift into gains and finally moves the optimum past k=5 (the page-1 weight-decay question, now urgent); adopt an early-stopped column for scoreboard entries at this scale; and run the same 4x/8x scale-up for rnn / lstm / attention to see whether recurrence overfits the same way at matched budget.

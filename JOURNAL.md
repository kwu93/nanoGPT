# Research Journal

Working log of experiments on the from-scratch models (`solo.py` lineage, `experiments/` framework).
Convention: when a research question comes up, log the proposed explanation here as a hypothesis with a falsifiable prediction *before* running the experiment, then record the result.

## Scoreboard (as of 2026-07-13, end of session)

Val NLL in nats/char on tiny Shakespeare (892k train / 223k val chars, vocab 65).
Standard setting: ~128 tokens/step, 5000 iters, dim_embed 32, dim_hidden 128, mean over 3 seeds; each model at its best measured context.
Full per-context grids are in the two comparison entries below.

| model | best at | val NLL | params |
|---|---|---|---|
| unigram table | no context | 3.328 | - |
| per-position MLP (`mlp`) | any (context-blind) | 2.51 | 15k |
| `mlp_sum` | 2 chars (degrades after) | 2.462 | 15k |
| `attention`, 1 layer, 32 dims | 8-16 chars (degrades at 32) | 2.218 | 17k |
| `mlp_concat` | 16 chars | 2.055 | 76k |
| `lstm` H=64 | 32 chars | 2.033 | 31k |
| `rnn` | 32 chars | **1.970** (best at <= 31k params) | 31k |
| `lstm` H=128 | 32 chars | 1.923 | 93k |
| `attention`, 6 layers x 128 dims | 32 chars | 1.913 | 1.2M |
| `attention`, 6 layers x 384 dims, lr 3e-4 | 32 chars | **1.883** (best single neural) | 10.7M |
| fixed n-gram table (a=0.01) | 4 chars (collapses to ln 65 by 32) | 1.876 | - |
| Witten-Bell backoff | 4 chars | 1.790 | - |
| Kneser-Ney backoff | 5 chars | **1.634** (best single model) | ~7.5M counts |
| `lstm` H=128 + KN, lam=0.25 | 32 chars + tables | **1.567** (best overall) | both |

## Cost accounting (storage / compute / data)

Parameter counts, description length, and training compute are three different budgets; winners flip depending on which is held equal (2026-07-13 discussion).

| model | stored numbers | ~size | train compute | tokens seen | val NLL |
|---|---|---|---|---|---|
| KN backoff, depth 8 | 7.5M counts | ~51 MB | ~16M dict ops (~1e-4 TFLOPs) | 892k, one pass | 1.634 |
| `rnn` | 31k floats | 124 KB | 0.12 TFLOPs | 640k | 1.970 |
| `lstm` H=128 | 93k floats | 372 KB | 0.36 TFLOPs | 640k | 1.923 |
| `attention` 6L x 128d | 1.2M floats | 4.8 MB | 4.6 TFLOPs | 640k | 1.913 |
| `attention` 6L x 384d | 10.7M floats | 43 MB | 41 TFLOPs | 640k | 1.883 |

Readings: raw counts overstate both sides (KN's singletons are shrunk to ~zero effective DOF by discounting; SGD-at-5000-iters is implicit early stopping), and KN's tables are close to a reorganized copy of the corpus - the MDL sense in which tables memorize rather than learn.
At roughly equal description length (43 vs 51 MB) the big transformer still trails KN while spending ~400,000x the compute - but KN is at convergence by construction and every neural number is an at-budget snapshot of a still-falling curve.
At equal compute, KN wins absurdly; at equal storage with compression-aware accounting, the small recurrent models are the honest champions; at equal tokens with compute unbound, the transformer's trajectory is the bet.

## Key learnings so far

1. Loss is bounded by the information the architecture lets a prediction *access*, not by parameters: the per-position MLP is pinned at the bigram floor because no operation mixes across positions, and capacity/context sweeps are flat by construction (structural zeros in the Jacobian cannot be trained away).
2. Every context-mixing scheme is h_t = sum over past of (factor x embedding); the factor's form is the architecture: tied weights (sum) loses order, per-offset weights (concat/conv) fixes order structurally, content-dependent scalars (attention) add dynamic selection.
3. Order in a 2-char window is worth ~0.2 nats, measured twice independently (bag-vs-ordered table floors: 0.200; sum-vs-concat neural gap: 0.205).
4. Summing raw embeddings actively hurts as the window grows (k>=4 is worse than no context): recency dilution plus superposition interference - information present in the input is useless if the architecture cannot separate it.
5. Count tables die of context sparsity (bag-8: 64% singleton contexts, 34% of val contexts novel); back-off fixes most of it and discounting fixes most of the rest (fixed table 2.88 -> WB 2.01 -> KN 1.65 at n=8). Singleton over-trust was the dominant residual, per the confirmed KN prediction.
6. Well-engineered classical baselines are strong: KN (1990s counting) beats our best neural model by 0.53 nats at this scale, and would trail a 6-layer transformer by only ~0.13. Weak baselines flatter; strong ones keep you honest.
7. Methodology that paid off: pre-registered falsifiable predictions (wrong ones were the most informative), fixed seeds (bit-identical cross-model checks at k=1), floors/ceilings computed from data as anchors, and normalization sanity checks before trusting new probability models.
8. Attention is not magic at tiny scale: one layer is a selective sum (window averaged into one vector before the FFN), and at 32 dims it loses to concat and rnn at every context length, degrading 16 -> 32 as diffuse softmax drifts toward the sum pathology. Depth 2-4 narrows but does not close the gap; bigram.py's win at the same token budget uses 384 dims, pointing at width as the missing variable (untested).
9. Position-averaged loss hides a warm-up effect: position t has only t+1 chars of effective context, so seq_len=S really means mean context (S+1)/2. Short-seq cells are dominated by context-starved positions; compare against position-averaged floors or mask warm-up positions.
10. Supervision is a first-order variable: raising tokens/step 32 -> 128 was worth ~0.10 nats to the rnn and cut seed variance ~15x, and with equal supervision the neural bigram sits 0.002 off its exact information floor. Attribute gaps to architecture only after matching tokens/step.
11. Gating costs 4x parameters per unit of state width: at equal params the LSTM loses to the vanilla rnn (state width binds), at equal width it wins. "X beats Y" needs a stated matching - params, width, storage, or compute - because the winner flips with the matching.
12. Width unlocks attention (2.29 -> 1.88 with scale) but tokens cap everything: at 640k supervised tokens no neural model catches KN's one full counting pass. Meanwhile interpolating the LSTM with KN (lam=0.25) gives 1.567 - complementary errors, and the cheapest jump past both components.

## Reflections

The session's arc, compressed: "why doesn't capacity matter?" turned out to be the wrong first question - the right ladder was information access (what can the prediction see), then mechanism (how is it combined: sum < fixed offsets < composition, with dynamic selection pending scale), and only then capacity.
Counting-based floors (n-gram, bag, backoff) were the single most valuable instrument: every neural result got its meaning from the distance to a floor, not from its absolute value.
Pre-registration earned its keep in refutations: WB's non-monotonicity exposed singleton over-trust, and attention missing its bands at every step of the rematch is what isolated width as the live hypothesis.
Open questions carried forward: the width fan-out sweep on attention at seq_len=32 (the original question, finally answerable); training-budget scaling (everything neural is at-budget, not at-convergence); modified KN for a tighter classical floor; warm-up masking for cleaner comparisons.

## 2026-07-13: Loss vs iterations - which at-budget rankings survive convergence?

**Question.**
Every neural scoreboard number is a 5000-iteration snapshot of a still-falling curve, while KN is converged by construction.
Which rankings invert with 4x and 16x more training, and does any neural model pass KN (1.634)?

**Method.**
Five models at seq_len=32, ~128 tokens/step: `mlp_concat` k=32, `rnn` H=128, `lstm` H=128, `attention` 6L x 128d (lr 1e-3), `attention` 6L x 384d (6 heads, lr 3e-4, MPS).
num_iters in {5k (existing, 3 seeds), 20k, 80k} - 80k = 10.2M tokens ~= 11.5 epochs.
New runs: seed 0 only (seed std at this supervision was 0.003-0.02); eval_every scaled to num_iters/20 so every curve has 21 points; no LR schedule, no dropout (deliberately vanilla - regularization is a future entry).
Curves land in runs.jsonl, so both final and minimum val loss are recoverable per run.

**Prediction (pre-registered).**
1. No rank inversions among the five at 80k (same order as 5k, gaps wider), except concat converges early: flat from 20k to 80k at ~2.0.
2. The 10.7M transformer passes KN: min val in [1.50, 1.62] by 80k - the neural-beats-classical crossover happens at ~16x our standard budget.
3. Overfitting appears where capacity/data is worst: the 384d model's val curve turns up before 80k (min < final); the small recurrent models do not turn up.
4. The LSTM's edge over the rnn grows with training (>= 0.06 at 80k, from 0.047 at 5k): gating's trainability advantage compounds.
5. On the compute axis the ranking flips: lstm at 80k (~5.8 TFLOPs) beats attention-384 at 5k (41 TFLOPs) by at least 0.05 - small-model-trained-longer beats big-model-trained-short at 7x less compute.

**Result.**
Pending.

**Conclusion / next steps.**
Pending.

## 2026-07-13: Does width unlock attention? (the fan-out experiment, at last)

**Question.**
Tiny attention lost to everything; bigram.py wins with the same token budget but 384 dims.
Is width the variable that makes attention work - the capacity fan-out the journal's first entry went looking for?

**Method.**
seq_len=32, batch 4 (~128 tokens/step), 3 seeds, on MPS (first GPU use; ~10 min per heavy run).
Width grid at lr 1e-3, n_heads=4, dim_hidden = 4 x dim_embed: n_layers=1 with dim_embed {64, 128, 384}; n_layers=6 with dim_embed {64, 128}.
Plus the bigram.py replication point: n_layers=6, dim_embed=384, n_heads=6, lr 3e-4 (10.7M params).

**Prediction (pre-registered).**
1. Width finally fans out: 1-layer val loss improves monotonically 2.29 (32d) -> below 2.0 by 384d.
2. 6 layers x 128d lands in [1.75, 1.95], beating the rnn (1.970) - the first neural model to do so.
3. The replication point (6L x 384d, lr 3e-4) reaches [1.45, 1.65], beating KN (1.634) and setting a new overall best.
4. Depth and width interact: 6L x 64d beats 1L x 384d despite fewer parameters (composition plus capacity beats capacity alone).

**Result.**
seq_len=32, ~128 tokens/step, 3 seeds, MPS (first GPU runs in the harness):

| config | params | val loss |
|---|---|---|
| 1L x 64d | 60k | 2.081 |
| 1L x 128d | 219k | 1.986 |
| 1L x 384d | 1.84M | 1.974 |
| 6L x 64d | 309k | 1.971 |
| 6L x 128d | 1.21M | 1.913 |
| 6L x 384d (6 heads, lr 3e-4) | 10.7M | **1.883** |

Prediction 1 confirmed: width finally fans out at 1 layer (2.291 -> 2.081 -> 1.986 -> 1.974), though saturating by 384d.
Prediction 2 confirmed: 6L x 128d (1.913) beats the rnn (1.970) - the first attention configuration to do so.
Prediction 3 REFUTED: the replication point reaches 1.883, not [1.45, 1.65], and does not beat KN (1.634).
Post-hoc: the "~1.5 at the same budget" reference conflated settings - Karpathy's ~1.48 comes from block_size=256, batch 64, dropout, LR schedule, and far more tokens; at 640k supervised tokens (< 1 epoch) the 10.7M-param model is deeply data/budget-limited.
Prediction 4 confirmed in direction, margin within noise: 6L x 64d (1.971, 309k params) matches 1L x 384d (1.974, 1.84M params) with 6x fewer parameters.

**Conclusion / next steps.**
Width was the missing variable for attention - capacity fans out at last, answering the journal's original question in the affirmative once the architecture can route information - but scale without tokens cannot catch counting: every neural point is budget-capped, and the gap to KN (0.25 nats) is now clearly a data/compute story, not an architecture story.
Composition remains cheaper than capacity (6L x 64 ~= 1L x 384 at 1/6 the params).
Next: train-longer curves (loss vs iterations per architecture) to see which at-budget rankings invert at convergence.

## 2026-07-13: Are neural and Kneser-Ney errors complementary? (interpolation)

**Question.**
KN (1.634) and the best neural model make their nats in different places - KN by exact short-context counts, neural by generalization.
Does interpolating them, p = lam * p_neural + (1 - lam) * p_KN, beat both components - the way neural LMs were actually deployed for their first decade?

**Method.**
Components: KN at depth 8, and the best neural model at seq_len=32 (rnn or lstm, whichever wins the gating entry), retrained at seed 0.
Neural probabilities computed over the val stream in consecutive 32-char windows (warm-up positions included, consistent with training eval); KN probabilities per position from the full stream.
NLL(lam) reported over a 21-point grid; both the best lam and the untuned lam=0.5 quoted, since selecting lam on val is a (mild, one-parameter) selection effect.

**Prediction (pre-registered).**
1. Even untuned lam=0.5 beats both components.
2. The best mixture reaches <= 1.62: a new overall best (unless the width entry's replication point got there first).
3. Optimal lam in [0.3, 0.6] - both components carry real weight, confirming complementarity rather than dominance.

**Result.**
Component selected by the gating entry: lstm H=128 (retrained seed 0, val 1.926; stream-eval NLL 1.921, consistent).
KN component at depth 8: 1.659.
NLL(lam) is a smooth U-curve: 1.659 (lam=0) -> 1.567 (lam=0.25) -> 1.592 (lam=0.5) -> 1.921 (lam=1).

Prediction 1 confirmed: untuned lam=0.5 (1.592) beats both components.
Prediction 2 confirmed: best mixture 1.5666 at lam=0.25 - new overall best, 0.067 below standalone KN's 1.634.
Prediction 3 REFUTED, narrowly: optimal lam is 0.25, just under the predicted [0.3, 0.6]; the neural model earns a real but junior share.

**Conclusion / next steps.**
The errors are genuinely complementary: a model 0.29 nats worse than KN still improves it by 0.07 when mixed, meaning the LSTM knows things the tables cannot represent (and vice versa - KN's exact counts anchor the mixture, hence the junior lam).
This reproduces the actual deployment recipe of 2000s-era neural LMs.
Caveat logged: lam selected on val (one parameter; the lam=0.5 row is the untouched reference).
Follow-up idea: per-position analysis of where the mixture's gains concentrate (prediction: positions where KN backs off below depth 3).

## 2026-07-13: Gating vs plain recurrence (LSTM)

**Question.**
The vanilla rnn folds input into state through one fixed tanh; the LSTM computes input-dependent multiplicative gates (forget/input/output) around an additive cell-state highway.
Is gating worth nats at equal parameters and budget?

**Method.**
`lstm` registered in `experiments/model.py` (hand-rolled fused-gate cell).
seq_len=32, batch 4, 5000 iters, 3 seeds.
Two sizes: dim_hidden=64 (31.1k params - matched to the rnn's 31.1k almost exactly) and dim_hidden=128 (92.9k, width-matched to the rnn's hidden size).

**Prediction (pre-registered).**
1. Param-matched (H=64): ties or beats the rnn, in [1.87, 1.97].
2. Width-matched (H=128): clearly beats it, in [1.82, 1.93] - new best neural (pending the width entry).
3. Both remain well short of KN (1.634): gating improves optimization and routing, but no recurrent state fully substitutes for exact short-context statistics at this budget.

**Result.**
seq_len=32, batch 4, 3 seeds:

| config | params | state size | val loss |
|---|---|---|---|
| rnn H=128 (reference) | 31.1k | 128 | 1.970 +/- 0.003 |
| lstm H=64 (param-matched) | 31.1k | 64 | 2.033 +/- 0.013 |
| lstm H=128 (width-matched) | 92.9k | 128 | 1.923 +/- 0.006 |

Prediction 1 REFUTED: the param-matched LSTM (2.033) loses to the rnn (1.970) by 0.063.
Gates cost 4x the weights per unit of state, so at equal parameters the LSTM's memory is half as wide (64 vs 128 dims) - and at this scale, state width beats routing.
Prediction 2 confirmed (barely - 1.923 vs the predicted ceiling 1.93): at equal state width, gating wins by 0.047, at 3x the parameters.
Prediction 3 confirmed: both far from KN.

**Conclusion / next steps.**
Gating is not free: it buys trainability and routing at a 4x parameter tax on state width, and the tax only pays when state width is not the binding constraint.
At 31k params the vanilla rnn remains the efficiency champion; the LSTM takes the (unmatched) neural recurrence crown at 1.923.
Worth revisiting at longer context / more training, where the vanishing-gradient advantage should grow.

## 2026-07-13: Attention joins the comparison (+ classical columns)

**Question.**
Where does attention land on the architecture-vs-context grid, and how does the whole neural family compare against the classical models (fixed n-gram table at alpha=0.01, Kneser-Ney back-off) at matching context?

**Method.**
`attention` registered in `experiments/model.py`: bigram.py in miniature - token + position embeddings, n_layers (default 1) of pre-LN causal multi-head attention (default 4 heads) + FFN with residuals, at baseline width (~17-18k params).
Same grid as the previous entry: seq_len in {1, 2, 3, 4, 8, 16, 32}, ~128 tokens/step, 3 seeds.
Classical columns computed exactly at each context length: fixed (S+1)-gram table with alpha=0.01 (bytes-keyed for the big contexts), KN at depth min(S, 8) (deeper levels are memory-prohibitive and near-weightless; the cap is noted where it binds).

**Prediction (pre-registered).**
1. Fixed table (a=0.01) collapses at long context: ~3.2 at 8 chars, ~4.0 at 16, ~4.17 (= ln 65, the all-unseen ceiling) at 32. KN stays saturated at ~1.65-1.67 from 4 chars on.
2. Attention at seq_len=1 ~= 2.51 (degenerate self-attention = per-position MLP); at seq_len <= 4 within ~0.02 of concat/rnn (window saturation makes mechanism invisible).
3. Attention improves monotonically through 32 with no concat-style flattening (dynamic weights can ignore uninformative distant tokens): [1.97, 2.05] at 16, [1.92, 2.04] at 32.
4. Riskiest call: 1-layer attention does NOT beat the rnn at seq_len=32 (within 0.05 either side of 1.970). One round of mixing reaches everything but composes nothing; the rnn composes 32 nonlinear steps. If attention loses, the rematch is n_layers=2.
5. Attention has the fewest parameters of any context-aware model at 32 (~18k vs rnn 31k, concat 142k), roughly flat in seq_len.

**Result (round 1: n_layers=1).**
Attention (mean of 3 seeds; max std 0.023) vs the prior grid:

| seq_len | attention | rnn | mlp_concat | fixed table a=0.01 | KN |
|---|---|---|---|---|---|
| 1 | 2.509 | 2.510 | 2.506 | 2.509 | 2.504 |
| 2 | 2.361 | 2.337 | 2.336 | 2.132 | 2.100 |
| 3 | 2.311 | 2.248 | 2.247 | 1.885 | 1.796 |
| 4 | 2.264 | 2.185 | 2.185 | 1.876 | 1.654 |
| 8 | 2.218 | 2.070 | 2.088 | 3.253 | 1.659 |
| 16 | 2.218 | 2.003 | 2.055 | 4.142 | 1.659* |
| 32 | 2.291 | 1.970 | 2.059 | 4.174 | 1.659* |

*KN capped at depth 8; fixed table computed exactly at each context (4.174 = ln 65: every context unseen).

Prediction 1 confirmed precisely (fixed-table collapse to the all-unseen ceiling; KN saturation).
Prediction 4 confirmed directionally but not in magnitude: 1-layer attention loses to the rnn at seq_len=32 by 0.32, far outside the predicted +/- 0.05 band.
Predictions 2 and 3 REFUTED: attention trails concat/rnn from seq_len=3 on (0.06-0.08 at 3-4), and it is non-monotone, degrading from 16 to 32 (2.218 -> 2.291, ~3x seed std).
Prediction 5 confirmed (17.9k params at 32).

Post-hoc diagnosis: a single attention layer is a *selective sum* - the window is compressed to one dim_embed vector by a softmax-weighted average before the FFN sees it.
Unlike concat (which hands the FFN all k x C dims) and the rnn (which composes sequentially), one round of mixing is lossy per position; and when undertrained softmax weights stay diffuse, a wide window drifts toward the mlp_sum pathology (consistent with the 16 -> 32 degradation and with 2.29 sitting between concat 2.06 and sum 3.00).
Depth should fix both failure modes: layer 2 can read features already mixed by layer 1 (composition), and more training sharpens the softmax (selection).

**Rematch prediction (pre-registered): n_layers in {2, 4} at seq_len=32.**
1. n_layers=2 (~30.6k params, matching the rnn's 31k almost exactly): improves to 2.05-2.15 but still behind the rnn (1.970).
2. n_layers=4 (~56k): 1.95-2.10, at best matching the rnn at this budget.

**Result (rematch).**
At seq_len=32, 128 tokens/step, 3 seeds:

| model | params | val loss |
|---|---|---|
| attention, 1 layer | 17.9k | 2.291 +/- 0.023 |
| attention, 2 layers | 30.5k | 2.238 +/- 0.013 |
| attention, 4 layers | 55.7k | 2.174 +/- 0.019 |
| rnn (reference) | 31.1k | 1.970 +/- 0.003 |

Both rematch predictions REFUTED: 2 layers lands at 2.238 (predicted 2.05-2.15) and 4 layers at 2.174 (predicted 1.95-2.10).
Depth buys a steady but slow ~0.05-0.06 per doubling; even 4 layers trails concat (2.059), and the parameter-matched comparison is stark: 2-layer attention (30.5k params) loses to the rnn (31.1k) by 0.27 nats.

**Conclusion / next steps.**
At this scale and budget, attention is the *worst* context-aware neural mechanism, despite being the one that wins at real scale.
The recurrent inductive bias (recency via composition) comes for free; attention must learn what to look at from scratch through diffuse softmax weights and position embeddings, and 5000 iters at 32 dims is not enough.
The outstanding anomaly makes the missing variable conspicuous: bigram.py reaches ~1.5 with the SAME token budget (5000 iters x 128 tokens/step) but 384 dims / 6 heads / 6 layers / 10.8M params - suggesting width (head dim, embedding capacity), not depth, is what our attention lacks.
This finally reconnects to the very first journal entry: the dim_embed x dim_hidden fan-out sweep, now on a model that can use capacity.
Next experiment: width sweep on attention at seq_len=32 (dim_embed 32 -> 384), watching for the capacity fan-out the per-position MLP could never show; note bigram.py also differs in lr (3e-4 AdamW vs our 1e-3 Adam) - worth controlling.

## 2026-07-13: All architectures vs context length (the full comparison)

**Question.**
How do all four neural architectures (`mlp`, `mlp_sum`, `mlp_concat`, `rnn`) scale with context length across seq_len in {1, 2, 3, 4, 8, 16, 32}?

**Design.**
Context length and seq_len are tied per model: `mlp` sees 1 char regardless, `mlp_sum`/`mlp_concat` get context_k = seq_len, `rnn`'s context is seq_len itself.
Tokens per step are held at ~128 for every cell (batch_size = 128 // seq_len; 126 at seq_len=3) to remove the supervision confound quantified in the RNN entry.
3 seeds per cell, 84 runs, baseline width, 5000 iters.
Reference floors: KN backoff at matching context (2.504 at 1 char, 2.100 at 2, 1.796 at 3, 1.654 at 4; ~1.63 saturated beyond).

**Prediction (pre-registered).**
1. `mlp` is flat (2.49-2.55) at every seq_len: context is structurally 1 char, and supervision is now equalized.
2. `mlp_sum` peaks at seq_len=2 (~2.35-2.45) and degrades monotonically after, ending worst of all models at 32 (>= 2.7): recency dilution plus superposition.
3. `mlp_concat` improves through seq_len ~8 (to ~2.00-2.10), then goes flat or slightly worse by 32: distant offsets carry little signal but add per-offset weights (131k params at k=32) and noise.
4. `rnn` tracks `mlp_concat` within ~0.05 at seq_len <= 4, is strictly best at >= 16, and improves monotonically to ~1.97 at 32.
5. At seq_len=1 all four models agree within ~0.05 (same information, near-same architecture).
6. No neural model reaches the KN floor at matching context; gap >= 0.25 nats everywhere.

**Result.**
Val loss (mean over 3 seeds; max seed std 0.024 across all 28 cells):

| seq_len | mlp | mlp_sum | mlp_concat | rnn | concat params | rnn params |
|---|---|---|---|---|---|---|
| 1 | 2.506 | 2.506 | 2.506 | 2.510 | 14.7k | 31.1k |
| 2 | 2.510 | 2.462 | 2.336 | 2.337 | 18.8k | 31.1k |
| 3 | 2.528 | 2.508 | 2.247 | 2.248 | 22.9k | 31.1k |
| 4 | 2.515 | 2.537 | 2.185 | 2.185 | 27.0k | 31.1k |
| 8 | 2.530 | 2.704 | 2.088 | 2.070 | 43.4k | 31.1k |
| 16 | 2.525 | 2.871 | 2.055 | 2.003 | 76.1k | 31.1k |
| 32 | 2.526 | 3.002 | 2.059 | **1.970** | 141.7k | 31.1k |

Predictions 1, 2, 4, 5 confirmed; 3 confirmed with the plateau arriving at 16 rather than 8; 6 REFUTED at short context.
Details:
1. `mlp` flat at 2.506-2.530 across a 32x context range.
2. `mlp_sum` peaks at seq_len=2 (2.462), degrades monotonically to 3.002, worst model from seq_len=4 on.
3. `mlp_concat` improves through 16, then a within-noise uptick at 32 (+0.004).
4. `rnn` tracks concat to within 0.003 at seq_len <= 4 (predicted 0.05 - the agreement is nearly exact), pulls ahead from 8, and wins clearly at 16 (0.051) and 32 (0.089) with 4.6x fewer parameters than concat at 32.
5. All four models within 0.0045 of each other at seq_len=1.
6. At seq_len=1 the gap to the KN floor is 0.002, not >= 0.25: with 128 tokens/step the neural bigram converges essentially to its information floor. The earlier 0.03-0.05 bigram-ceiling gap was undertraining, not architecture.

Subtlety surfaced by an apparent anomaly: concat k=2 here (2.336, seq_len=2) looks worse than the earlier concat k=2 run (2.244, seq_len=8) despite 4x more supervision.
Explanation: the loss averages over positions, and position t only has t+1 characters of effective context, so at seq_len=2 half of all predictions are context-starved warm-up positions vs 1/8 at seq_len=8.
Mean effective context at seq_len=S is (S+1)/2, identical for all models (which is why concat and rnn still match), but it means naive "neural at context S vs KN at depth S" comparisons overstate the neural handicap.
Against position-averaged KN floors (e.g. ~2.30 at seq_len=2, ~1.70 at 32), the neural models sit 0.03-0.34 above, not 0.24-0.53.

**Conclusion / next steps.**
Mechanism is irrelevant when the window is small: per-offset weights and composed state land on identical losses at seq_len <= 4, because both capture everything a short window offers.
Mechanism decides everything at scale: shared-weight composition (rnn) keeps improving where per-offset weights (concat) flatten and order-blind addition (sum) actively collapses - and it does so with constant parameters.
The context axis now cleanly separates all four architectures, answering the original fan-out question in mechanism form.
Future comparisons should either mask warm-up positions or quote mean effective context (S+1)/2.
Next: attention at seq_len=32, 128 tokens/step, 3 seeds - beat rnn 1.970, chase KN 1.634.

## 2026-07-13: Is an RNN the missing intermediate between concat and attention?

**Question.**
The architecture table so far: sum (shared weights, order lost), concat (per-offset weights, order structural, params grow with window), attention (shared weights, dynamic selection, order injected).
There is a hole: shared weights *and* order awareness with no parameter growth in context length.
Does a vanilla RNN fill it, and where does it land on the scoreboard?

**Hypothesis.**
The RNN (h_t = tanh(Wx e_t + Wh h_{t-1}), registered as `rnn`) processes the window sequentially through one shared cell.
Like `mlp_sum` it reuses the same weights for every offset, but it *composes* instead of *adds*: function composition is non-commutative, so order survives structurally.
The hidden state is a learned fixed-size summary of the entire prefix, so context is unbounded in principle (capped at `seq_len` here because batches are random crops with h_0 = 0) and the parameter count is independent of context length.
Historically this is the step that displaced KN n-grams (Mikolov 2010); its own known weaknesses - the fixed-size state bottleneck and vanishing gradients through tanh chains - are what attention later removed.

**Prediction (pre-registered).**
Baseline width, 5000 iters, batch 4, 3 seeds, `seq_len` in {8, 32}:
1. At seq_len=8 the RNN lands in the concat k=8 band (2.10-2.35): same accessible information, different mechanism, slightly harder optimization.
2. It beats `mlp_sum` k=8 (2.764) by >= 0.4 nats at identical parameter sharing - the direct demonstration that composition preserves what addition destroys.
3. Parameter count is identical at seq_len 8 and 32 (~31k), unlike concat (which would need ~131k input weights at k=32).
4. seq_len 8 -> 32 improves val loss by < 0.1 nats at this budget: vanishing gradients plus undertraining limit the *effective* context well below 32. Confound noted: T=32 also supervises 4x more tokens per step, so any gain is an upper bound on the context effect.

**Result.**
Run 2026-07-13, 3 seeds each, appended to `runs.jsonl` (params 31,073 in every configuration):

| config | tokens/step | val loss (mean +/- std) |
|---|---|---|
| seq_len=8, batch=4 | 32 | 2.1745 +/- 0.0375 |
| seq_len=8, batch=16 (control) | 128 | 2.0704 +/- 0.0123 |
| seq_len=32, batch=4 | 128 | **1.9695 +/- 0.0025** |

Prediction 1 confirmed: seq_len=8 RNN (2.174) lands within noise of concat k=8 (2.180) - same accessible information, same loss, different mechanism.
Prediction 2 confirmed: 0.59 nats better than `mlp_sum` k=8 (2.764) with the same weight sharing; composition preserves what addition destroys.
Prediction 3 confirmed: parameter count identical at seq_len 8 and 32.
Prediction 4 REFUTED: seq_len 8 -> 32 improved val loss by 0.205 nats, double the predicted < 0.1.
The batch=16 control decomposes the gain almost exactly in half: 0.104 from more supervision (32 -> 128 tokens/step at fixed context) and 0.101 from context alone (8 -> 32 chars at fixed tokens/step).
So characters 9-32 back are worth ~0.10 nats to a vanilla RNN at this budget - the tanh state carries usable information much further back than the vanishing-gradient intuition suggested.
Also notable: seed variance collapses with more tokens/step (std 0.0375 -> 0.0025).

**Conclusion / next steps.**
The RNN completes the architecture table: shared weights AND order awareness AND zero parameter growth in context length - the first model in the series where extending context is architecturally free.
New best neural: 1.9695, still 0.34 nats behind KN (1.634).
The 0.10-nat supervision effect is a reminder that our 5000-iter budget undertrains everything; scoreboard comparisons are at-budget, not at-convergence.
Next: attention, target 1.634; also worth running mlp_concat and attention at the seq_len=32 / batch=4 setting for a fair three-way mechanism comparison at equal tokens/step.

## 2026-07-13: Kneser-Ney vs the singleton diagnosis

**Question.**
Witten-Bell backoff softened but did not eliminate the sparse-table U-turn (best 1.790 at n=5, rising to 2.010 by n=8).
The post-hoc diagnosis: WB over-trusts singletons, because a context seen once with one continuation gets lam = 0.5.
Kneser-Ney's absolute discounting subtracts a flat D (typically ~0.75) from every count, capping a singleton's contribution at max(1-D, 0) ~= 0.25 before interpolation, and its lower orders use continuation counts (how many distinct contexts a character completes) instead of raw frequency.
If the singleton diagnosis is right, KN should fix the non-monotonicity that WB could not.

**Method.**
`KneserNeyModel` in `experiments/ngram.py`: interpolated KN, one discount per level estimated by the standard D = n1 / (n1 + 2 n2) rule, lower orders built on continuation counts derived from the next-higher raw table.
Correctness check before trusting any NLL: probabilities must sum to 1 over the vocabulary for sampled contexts at several depths.

**Prediction (pre-registered).**
1. The KN val ladder is monotone non-increasing through n=8 (up to ~0.01 slack); the n=6-8 U-turn disappears. This is the direct test of the singleton diagnosis; if the ladder still turns, the diagnosis was wrong or incomplete.
2. New best table baseline at or below WB's 1.790, expect 1.70-1.78, and reached at n>=6 rather than n=5.
3. At n=2-3 KN ~= WB ~= fixed tables (dense-count regime; continuation counts matter little at character level where every char appears in many contexts).

**Result.**
Normalization check passed first (worst |sum(p) - 1| = 7.8e-16 over 160 sampled context/depth pairs).
Val NLL ladder:

| n | WB backoff | Kneser-Ney |
|---|---|---|
| 2 | 2.504 | 2.504 |
| 3 | 2.105 | 2.100 |
| 4 | 1.866 | 1.796 |
| 5 | 1.790 | 1.654 |
| 6 | 1.828 | **1.634** |
| 7 | 1.913 | 1.643 |
| 8 | 2.010 | 1.653 |

Prediction 1 largely confirmed: the U-turn collapses from +0.220 nats (WB, n=5 to 8) to +0.019 (KN, n=6 to 8) - a 10x reduction, though the residual uptick slightly exceeds the pre-registered 0.01 slack.
Singleton over-trust was the dominant cause of the non-monotonicity, but not quite the whole story; modified KN (separate discounts for count-1/2/3+ n-grams) is the known refinement for the remainder.
Prediction 2 confirmed and exceeded: best 1.634 at n=6, below the predicted 1.70-1.78 band and 0.156 better than WB's best.
Prediction 3 confirmed: n=2 identical to WB (2.504), n=3 within 0.005 - discounting and continuation counts only matter once counts get sparse, exactly as reasoned.

**Conclusion / next steps.**
The singleton diagnosis was correct and mechanistically specific: capping one-observation evidence (absolute discounting) plus versatility-based back-off (continuation counts) is what fixed the high-order ladder.
The classical baseline moves from 1.790 to **1.634** (KN, n=6).
Scoreboard implications: mlp_concat k=4 (2.164) is now 0.53 nats behind a count-based method, and the bigram.py transformer's expected ~1.5 would beat the best classical table by only ~0.13 nats on this 892k-char dataset - the neural advantage at this scale is real but modest, and grows with data and context length.
Next: the attention entry, with 1.634 as the number to beat; the dim fan-out sweep on a context-aware model is still standing.

## 2026-07-13: Does back-off fix the sparse-table U-turn?

**Question.**
The ordered n-gram ladder turns around past n~=5 (and the bag ladder past k~=2) because fixed-context tables spread the data over combinatorially many contexts: at bag-8, 64% of contexts are singletons and 34.5% of val contexts are entirely novel.
Classical NLP answers with back-off / interpolation: use the longest context that has evidence and retreat gracefully when it doesn't.
Does that eliminate the U-turn on this data, and where does the table baseline end up?

**Method.**
Added `BackoffModel` to `experiments/ngram.py`: Witten-Bell interpolation, p_k = lam * ML_k + (1 - lam) * p_{k-1} recursing down to uniform, with lam(ctx) = c(ctx) / (c(ctx) + d(ctx)) where d counts distinct continuations of ctx.
Chosen over Katz (hard back-off only on zero counts) for simplicity, and over Kneser-Ney (the classic SOTA) because Witten-Bell is parameter-free, which also removes the alpha-tuning wart from the earlier ladder entries.

**Prediction (pre-registered).**
1. The val ladder becomes monotone non-increasing in n: each added level can only contribute where it has evidence, so more context never hurts.
2. New best table baseline, below the fixed-table best of 1.876 (5-gram, a=0.01); expect roughly 1.70-1.82 at n=8.
3. Gains saturate around n~=5-6: 892k training characters cannot populate longer contexts often enough for lam to trust them.

**Result.**
Val NLL by max context length (fixed table best alpha vs Witten-Bell backoff):

| n | fixed table (best alpha) | WB backoff |
|---|---|---|
| 2 | 2.500 | 2.504 |
| 3 | 2.099 | 2.105 |
| 4 | 1.885 | 1.866 |
| 5 | 1.876 | **1.790** |
| 6 | 2.118 | 1.828 |
| 7 | 2.480 | 1.913 |
| 8 | 2.879 | 2.010 |

Prediction 2 confirmed: new best table baseline 1.790 at n=5, in the predicted 1.70-1.82 band.
Prediction 3 confirmed: gains stop at n=5.
Prediction 1 REFUTED: the ladder is not monotone.
Backoff softens the U-turn enormously (2.01 vs 2.88 at n=8) but does not eliminate it.

Post-hoc diagnosis: Witten-Bell over-trusts singletons.
For a context seen once with one continuation, lam = 1/(1+1) = 0.5, so a single observation still gets half the probability mass; at n=8 two-thirds of contexts are singletons, so the top level injects memorization noise faster than evidence.
Kneser-Ney exists precisely to fix this (absolute discounting caps every count's influence, and lower orders use continuation counts rather than raw frequencies); it should restore monotonicity and land somewhat lower.

**Conclusion / next steps.**
Back-off largely solves sparsity and sets the real classical baseline at ~1.79 val NLL, which reframes the scoreboard: our best neural model so far (mlp_concat k=4, 2.164) is 0.37 nats WORSE than a count-based method from the 1990s.
Tiny undertrained MLPs do not beat well-engineered tables; the neural payoff requires more training budget and better context use.
Targets for the attention model: beat 1.790 (bigram.py's 6-layer transformer reportedly trains to ~1.5, i.e. below any table).
Optional side quest: implement Kneser-Ney to test the singleton diagnosis.

## 2026-07-13: What is order worth? Summing the window vs concatenating it

**Question.**
`mlp_concat` mixes positions by concatenation: h = sum_j W_j e(x[t-j]), a separate weight block per offset.
The minimal alternative is summation, `mlp_sum`: h = W sum_j e(x[t-j]), all offsets tied to one W.
Algebraically this is the concat model with W_j forced equal, and a sum of embeddings is permutation-invariant, so the model sees only the *multiset* (bag) of the last `context_k` characters - it cannot even tell which character is most recent.
How much of the concat gains were order information?

**Background: bag-of-k table floors.**
Extended `experiments/ngram.py` with a `key=` hook; keying the table on the sorted window measures H(next | unordered bag of last k chars):

| k | bag contexts | val NLL (a=1) | val NLL (a=0.01) | ordered floor for reference |
|---|---|---|---|---|
| 1 | 65 | 2.500 | 2.509 | 2.500 (identical by definition) |
| 2 | 1,003 | 2.299 | 2.339 | 2.099 (trigram) |
| 4 | 24,203 | 2.471 | 2.327 | 1.885 (5-gram, a=0.01) |
| 8 | 342,201 | 3.665 | 3.479 | (table too sparse either way) |

Two striking features.
The unordered pair still beats the bigram (2.30 vs 2.50): for many character pairs order is nearly determined by the pair itself ({q,u}, {space, letter}), so a bag implicitly recovers much of the order.
And the bag floors are *non-monotone in k*: unlike ordered context, a bigger bag can be worse, because summing in more characters dilutes the identity of the most recent ones (the k=8 bag table is already worse than the bigram even before sparsity is fully to blame).

**Prediction (pre-registered).**
Baseline width, 5000 iters, batch 4, 3 seeds, `context_k` in {1, 2, 4, 8}:
1. k=1: identical to `mlp_concat` k=1 (~2.50-2.57); summing one element is concatenating one element.
2. k=2: lands near but above the bag-2 floor of 2.299, therefore clearly above `mlp_concat` k=2 (2.244), which already beats the sum model's *theoretical optimum*. The sum-vs-concat gap at k=2 measures the value of order in a two-character window.
3. Non-monotone in k, unlike concat: k=8 should be *worse* than k=2-4 because recency information is diluted (the neural model generalizes better than the sparse k=8 table, so expect ~2.4-2.8 rather than 3.5).
4. Parameter count is flat in k (~14.7k for all k), vs concat growing to 43k at k=8.

**Result.**
Run 2026-07-13, 3 seeds each, appended to `runs.jsonl`:

| context_k | mlp_sum | mlp_concat (prior entry) | bag floor | ordered floor |
|---|---|---|---|---|
| 1 | 2.5329 +/- 0.0298 | 2.5329 +/- 0.0298 | 2.500 | 2.500 |
| 2 | 2.4487 +/- 0.0364 | 2.2439 +/- 0.0394 | 2.299 | 2.099 |
| 4 | 2.6452 +/- 0.0464 | 2.1636 +/- 0.0315 | ~2.33 | ~1.89 |
| 8 | 2.7635 +/- 0.0512 | 2.1801 +/- 0.0298 | (sparse) | ~1.88 |

Prediction 1 confirmed in the strongest form: the k=1 seeds are bit-identical to `mlp_concat` k=1 (same module shapes, same seed, same batches), 2.5069/2.5655/2.5263.
Prediction 2 confirmed: sum k=2 (2.449) sits above the bag-2 floor and above concat k=2; the measured order gap, 2.449 - 2.244 = 0.205 nats, matches the floor-derived value of order, 2.299 - 2.099 = 0.200, almost exactly.
Prediction 3 confirmed and stronger than predicted: degradation begins at k=4 (2.645), not k=8, and k>=4 is worse than the *no-context* baseline (2.544).
Prediction 4 confirmed: 14,689 params at every k.

One result exceeds the hypothesis: sum k=4 (2.645) is far above even its own bag-4 floor (~2.33).
In principle the multiset is recoverable from a sum of embeddings, so this is not an information limit; superimposing 4+ vectors in 32 dims creates interference that the MLP cannot disentangle within the training budget.
The sum representation impedes optimization beyond what it destroys information-theoretically.

**Conclusion / next steps.**
Order in a 2-character window is worth ~0.2 nats, and the neural measurement reproduced the table-derived number.
Summing raw embeddings is worse than useless at scale: each added character dilutes the recency signal (bags are non-monotone in k) *and* adds superposition interference, so more context makes the model worse - a clean demonstration that information present in the input is only useful if the architecture gives the model a way to separate it.
This sharpens why attention works as a sum: it sums *linearly transformed* values with content-dependent coefficients (most near zero after softmax, limiting interference), and injects position into the summands rather than hoping order survives.
Next: the attention head itself, starting from the standing fan-out question (dim sweep on a context-aware model).

## 2026-07-13: Can an MLP gain context awareness without attention?

**Question.**
The per-position MLP is pinned at the bigram ceiling because no operation mixes information across positions.
Attention is one fix, but is there a minimal MLP-only change that buys context, as an intuition-building middle step?

**Background: the n-gram ladder (`experiments/ngram.py`).**
A count-based n-gram table (predict each char from the previous n-1 via smoothed lookup) is the optimal-ish reference for any model restricted to n-1 characters of context.
Its val NLL forms a ladder of entropy floors on this data (nats/char, train split -> val split):

| n | contexts seen | train NLL (a=1) | val NLL (a=1) | val NLL (a=0.01) |
|---|---|---|---|---|
| 1 | 1 | 3.309 | 3.328 | 3.328 |
| 2 | 65 | 2.452 | 2.500 | 2.509 |
| 3 | 1,360 | 1.952 | 2.099 | 2.132 |
| 4 | 10,899 | 1.755 | 1.998 | 1.885 |
| 5 | 46,224 | 1.910 | 2.244 | 1.876 |
| 6 | 125,229 | 2.236 | 2.669 | 2.118 |
| 7 | 245,344 | 2.575 | 3.100 | 2.480 |
| 8 | 379,330 | 2.853 | 3.454 | 2.879 |

Note the U-shape: past n~=5 the table gets *worse* because long contexts are sparse in 892k training characters (memorization + smoothing mass dominate).
Tables cannot share statistical strength between similar contexts; that failure is the original motivation for neural LMs (Bengio et al. 2003).
Best table on this data: ~1.88 val at n=4-5 with light smoothing.

**Hypothesis.**
The minimal MLP-only fix is concatenation (Bengio-style neural n-gram, registered as `mlp_concat`):
embed the last `context_k` characters, concatenate the embeddings into one vector of size `context_k * dim_embed`, and predict the next character from that.
Concatenation mixes across positions structurally (each relative offset gets its own slice of the first Linear's weight matrix), so the receptive field is `context_k` by construction, and the model should descend the n-gram ladder as `context_k` grows.

**Prediction (pre-registered).**
At baseline width (`dim_embed=32, dim_hidden=128`), `seq_len=8`, 5000 iters, batch 4, 3 seeds, sweeping `context_k` in {1, 2, 4, 8}:
1. `context_k=1` matches the per-position MLP baseline (2.50-2.57): same information, so concatenation alone buys nothing.
2. `context_k=2` clearly breaks the bigram floor of 2.500, landing between it and the trigram floor of 2.10 (expect ~2.15-2.35 given the training budget).
3. Val loss decreases monotonically in `context_k` with diminishing marginal gains, and `context_k=8` stays well above the best table floor (~1.88), because 5000 iters x 32 targets/step = 160k supervised tokens is less than one epoch of the training data (optimization-limited, not capacity-limited).

**Result.**
Run 2026-07-13, 3 seeds each, baseline width, appended to `runs.jsonl`:

| context_k | val loss (mean +/- std) | params | nearest table floors |
|---|---|---|---|
| 1 | 2.5329 +/- 0.0298 | 14,689 | bigram 2.500 |
| 2 | 2.2439 +/- 0.0394 | 18,785 | between bigram 2.500 and trigram 2.099 |
| 4 | 2.1636 +/- 0.0315 | 26,977 | near trigram 2.099 |
| 8 | 2.1801 +/- 0.0298 | 43,361 | near trigram 2.099 |

Prediction 1 confirmed: `context_k=1` (2.533) matches the per-position MLP baseline (2.544); concatenation alone buys nothing.
Prediction 2 confirmed: `context_k=2` (2.244) decisively breaks the bigram floor of 2.500 that no amount of width could break, landing inside the predicted 2.15-2.35 band.
Prediction 3 partially confirmed: gains diminish and `context_k=8` stays far above the 1.88 table best, but the means are not strictly monotone; k=8 (2.180) is a hair *above* k=4 (2.164), a 0.017 gap well inside seed noise.
Under this budget, characters 5-8 back contribute nothing measurable, consistent with the ladder flattening past n~=4.

**Conclusion / next steps.**
Context awareness in an MLP requires no attention, only an operation whose output at position t reads other positions' representations; concatenation is the minimal such operation and immediately unlocked ~0.37 nats.
This confirms the 2026-07-12 conclusion from the other direction: the bigram ceiling was purely architectural.
The k=4 vs k=8 plateau is likely budget-limited (160k supervised tokens < 1 epoch); rerun with 10-20x iters before concluding distant context is worthless to this model class.
Next: re-run the dim_embed x dim_hidden sweep on `mlp_concat` at `context_k=8` to test the 2026-07-12 fan-out prediction, then contrast with attention, whose advantages over concatenation are dynamic (data-dependent) weighting and parameter sharing across offsets.

## 2026-07-12: Why is val loss flat across the dim_embed x dim_hidden sweep?

**Question.**
A 3x3 sweep over `dim_embed` in {16, 32, 64} and `dim_hidden` in {64, 128, 256} (3 seeds each, 27 runs) produced val losses in a band of 2.535-2.563 across a 6x range of parameter counts (6.5k-38k).
Why does capacity not matter?

**Hypothesis.**
The model is architecture-limited, not capacity-limited.
The MLP block is applied to each position independently (`self.block(tok + pos)`), so no information flows across positions.
Despite `seq_len=8` in the batch tensors, the prediction for the next character is a function of exactly one character: the model class is a (factorized) bigram model.
Its loss floor is therefore the conditional entropy H(next char | current char), which every config in the sweep already has enough capacity to reach.

**Evidence.**
- A count-based bigram table (normalized pair counts from the train split, Laplace smoothed) achieves val NLL 2.4996; the swept models land at 2.535-2.563, a few hundredths above the lookup-table optimum. Unigram entropy is 3.309, so all models clearly do use the one character they can see.
- The config spread (0.028) is smaller than the seed-to-seed std (~0.033), so with 3 seeds most pairwise differences are noise.
- The only surviving trend is `dim_embed`: 16 < 32 ~= 64 consistently across all `dim_hidden` levels. Explanation: the embedding-to-logits path is a low-rank factorization of the 65x65 bigram table, and rank 16 slightly underfits while rank 32 suffices. `dim_hidden` shows no trend at all.
- Train ~= val (2.49 vs 2.53): no overfitting even at 38k params, so the flatness is not data limitation either.
- Side observation: position embeddings are dead weight in this setup. Batches are random crops, so window position carries no information about the target.

**Prediction (pre-registered).**
If the model truly cannot use context, val loss is invariant to `seq_len`.
Concretely, at `dim_embed=32, dim_hidden=128` (baseline: val 2.5443 +/- 0.0350 over 3 seeds at `seq_len=8`):
1. `seq_len=1` should match the baseline within seed noise (~2.51-2.58). Confound control: `seq_len=1` at `batch_size=4` supervises 4 tokens/step vs 32 for the baseline, so run it at `batch_size=32` to equalize tokens per step.
2. `seq_len=32` should also match the baseline, despite 4x more context and more supervised tokens per step.
If instead `seq_len=1` is clearly worse, the model is extracting something from context and the hypothesis is wrong.

**Result.**
Run 2026-07-12, 3 seeds each, all other config at baseline (`dim_embed=32, dim_hidden=128`):

| seq_len | batch_size | val loss (mean +/- std) | runs |
|---|---|---|---|
| 1 | 32 | 2.5304 +/- 0.0280 | 2.5228, 2.5069, 2.5614 |
| 8 (baseline) | 4 | 2.5443 +/- 0.0350 | from original sweep |
| 32 | 4 | 2.5262 +/- 0.0068 | 2.5259, 2.5331, 2.5195 |

Both predictions confirmed.
A model with 1 character of context matches the 8-character baseline within seed noise (numerically it is even slightly better), and 32 characters of context buys nothing.
The hypothesis survives falsification: the per-position MLP extracts zero information from context beyond the current character.

**Conclusion / next steps.**
Val loss is pinned at the bigram ceiling (count-based table floor: 2.4996), so sweeps over capacity *and* context are both flat by construction.
The binding constraint is cross-position information flow, not parameters.
Next experiment: add position mixing (the attention block, or as a crude baseline flatten the window into the MLP input), then re-run the identical `dim_embed` x `dim_hidden` sweep.
Expected outcome: the grid fans out, `seq_len` starts to matter, and position embeddings stop being dead weight.

# Research Journal

Working log of experiments on the from-scratch models (`solo.py` lineage, `experiments/` framework).
Convention: when a research question comes up, log the proposed explanation here as a hypothesis with a falsifiable prediction *before* running the experiment, then record the result.

## Scoreboard (as of 2026-07-13)

Val NLL in nats/char on tiny Shakespeare (892k train / 223k val chars, vocab 65).
Neural runs: 5000 iters, batch 4, dim_embed 32, dim_hidden 128, mean over 3 seeds.

| model | context seen | val NLL |
|---|---|---|
| unigram table | none | 3.328 |
| per-position MLP (`mlp`) | 1 char | 2.544 |
| `mlp_sum` k=2 | bag of 2 | 2.449 |
| bigram table | 1 char | 2.500 |
| `mlp_concat` k=2 | 2 chars | 2.244 |
| `mlp_concat` k=4 | 4 chars | **2.164** (best neural so far) |
| Witten-Bell backoff n=5 | 4 chars | 1.790 |
| Kneser-Ney backoff n=6 | 5 chars | **1.634** (best classical; current target) |
| attention (`bigram.py`, 6 layers, 10.8M params) | 8 chars | ~1.5 expected, not yet measured here |

## Key learnings so far

1. Loss is bounded by the information the architecture lets a prediction *access*, not by parameters: the per-position MLP is pinned at the bigram floor because no operation mixes across positions, and capacity/context sweeps are flat by construction (structural zeros in the Jacobian cannot be trained away).
2. Every context-mixing scheme is h_t = sum over past of (factor x embedding); the factor's form is the architecture: tied weights (sum) loses order, per-offset weights (concat/conv) fixes order structurally, content-dependent scalars (attention) add dynamic selection.
3. Order in a 2-char window is worth ~0.2 nats, measured twice independently (bag-vs-ordered table floors: 0.200; sum-vs-concat neural gap: 0.205).
4. Summing raw embeddings actively hurts as the window grows (k>=4 is worse than no context): recency dilution plus superposition interference - information present in the input is useless if the architecture cannot separate it.
5. Count tables die of context sparsity (bag-8: 64% singleton contexts, 34% of val contexts novel); back-off fixes most of it and discounting fixes most of the rest (fixed table 2.88 -> WB 2.01 -> KN 1.65 at n=8). Singleton over-trust was the dominant residual, per the confirmed KN prediction.
6. Well-engineered classical baselines are strong: KN (1990s counting) beats our best neural model by 0.53 nats at this scale, and would trail a 6-layer transformer by only ~0.13. Weak baselines flatter; strong ones keep you honest.
7. Methodology that paid off: pre-registered falsifiable predictions (2 of 9 were wrong in informative ways), fixed seeds (bit-identical cross-model checks at k=1), floors/ceilings computed from data as anchors, and normalization sanity checks before trusting new probability models.

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

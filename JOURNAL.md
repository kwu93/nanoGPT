# Research Journal

Working log of experiments on the from-scratch models (`solo.py` lineage, `experiments/` framework).
Convention: when a research question comes up, log the proposed explanation here as a hypothesis with a falsifiable prediction *before* running the experiment, then record the result.

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

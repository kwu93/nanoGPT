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

At 16x budget (80k iters ~= 11.5 epochs, seed 0): `lstm` 1.713 (still falling), `attention` 6L x 128d 1.687, `attention` 6L x 384d 1.669 at its minimum (overfits after iter 52k).
KN's 1.634 remains unbeaten by every single model; the 1.567 mixture is still best overall.

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
13. Sixteen times the training did not dethrone counting: unregularized scale overfits 892k chars before reaching the KN floor (the 10.7M model bottoms at 1.669 at iter 52k, then rises; KN stands at 1.634). Beating tables needs regularization and longer context, not just tokens - and the low-compute frontier belongs to small recurrent models (lstm 1.713 at 5.7 TFLOPs vs transformer 1.669 at 657 TFLOPs).
14. Information-bound vs capacity-bound is a measured dissociation: across 15.7x parameters at fixed budget (context 8), the context-blind mlp moves 0.010 nats while every context-aware architecture gains 0.21-0.33 - and the mlp cannot even *memorize* past its floor (train stops at the train-split bigram entropy). Capacity only matters when the architecture can reach information it has not yet extracted.
15. The params axis re-ranks mechanisms just like the width and budget axes did: the rnn-concat gap (~0.05) is scale-stable from 3.9k to 60k (weight sharing's win is not a scarcity artifact), the lstm's gate tax amortizes and overtakes concat at 60k, and attention stays last at every tier - at short context, capacity alone rehabilitates gating but not dynamic selection.
16. Val NLL and sample quality dissociate in both directions: the best fixed 5-gram table by NLL (1.876) collapses to uniform white noise ~11 chars into generation (smoothing mass is an escape hatch to uniform, and novel contexts are absorbing: 46k seen of 65^4 possible), while the alpha=0 table with *infinite* val NLL samples beautifully. One-step NLL on gold histories cannot see compounding-error behavior; backoff is what makes counting robust in the generation regime.

## Reflections

The session's arc, compressed: "why doesn't capacity matter?" turned out to be the wrong first question - the right ladder was information access (what can the prediction see), then mechanism (how is it combined: sum < fixed offsets < composition, with dynamic selection pending scale), and only then capacity.
Counting-based floors (n-gram, bag, backoff) were the single most valuable instrument: every neural result got its meaning from the distance to a floor, not from its absolute value.
Pre-registration earned its keep in refutations: WB's non-monotonicity exposed singleton over-trust, and attention missing its bands at every step of the rematch is what isolated width as the live hypothesis.
Open questions carried forward: the width fan-out sweep on attention at seq_len=32 (the original question, finally answerable); training-budget scaling (everything neural is at-budget, not at-convergence); modified KN for a tighter classical floor; warm-up masking for cleaner comparisons.

## 2026-07-16: Sampling the ladder - does text quality track val NLL?

**Question.**
Every result so far scores models by val NLL on real contexts, i.e. one-step prediction with a gold history.
Generation is a different regime: the model consumes its own output, so one bad character can push the history off the data manifold (exposure bias).
What does sampled text look like at each rung of the ladder - uniform random, unigram, bigram, trigram, 5-gram (fixed table, plus Witten-Bell and Kneser-Ney backoff at the same depth) - and does the context-blind `mlp` (val 2.51) read like the bigram table (val 2.50) it is information-equivalent to?

**Method.**
New `experiments/sample.py` (pixi task `sample`).
One shared 8-char primer (the first speaker-turn boundary in the val split), one fixed seed, 400 generated chars per model, fresh rng per model.
Count models built on the train split: fixed tables at k = 0/1/2/4 chars of context with alpha = 0.01, the alpha = 0 (pure ML) 5-gram as a control, and WB / KN backoff at depth 4.
Convention reminder: "5-gram" = 4 chars of context (n = k+1).
Neural: the saved per-position `mlp` (models/mlp-e32-h128-s0.pt).
Each sample is labeled with the val NLL of the exact distribution sampled from, recomputed and sanity-checked against the ladder entries.

**Prediction (pre-registered).**
1. Texture tracks context length: random = uniform char soup (space is 1/65 ~= 1.5% of chars instead of the corpus's roughly 16%, so no word rhythm at all); unigram = word-length runs of unpronounceable letters (whitespace rhythm right, nothing else); bigram = pronounceable digraphs and accidental real 2-3 letter words; trigram = mostly pronounceable pseudo-words plus common real short words, first hints of the NAME-colon-newline speaker format; the 5-gram family = mostly real words, names, and correctly formatted speaker turns.
2. Headline risk prediction: the alpha = 0.01 5-gram table - the best fixed table by val NLL (1.876) - degenerates during generation. Per step it puts about (65 - d) * 0.01 / (c + 0.65) mass on unseen continuations; averaged over visited contexts that is roughly 0.63 * K / N ~= 3% per char (K ~= 46k seen contexts, N = 892k), so a 400-char sample almost surely (p > 0.999) steps off the table at least once. An escape usually lands in a novel 4-char context (only ~46k of 65^4 ~= 17.9M are seen), where the distribution is uniform and the successor contexts stay novel. Expect visible collapse into white noise, probably within the first ~100 chars, possibly after a recovered stumble or two.
3. The alpha = 0 ML control cannot leave seen support (a sampled continuation extends a seen 5-gram, whose 4-char suffix is again a seen context), so it stays Shakespeare-shaped for all 400 chars - the best-looking count sample, and the one closest to regurgitating the corpus. Its val NLL is infinite (novel val events get probability zero), the mirror image of prediction 2: NLL and sample quality can dissociate in both directions.
4. WB and KN are structurally immune to the collapse (a novel context hands its mass to shorter contexts, never to uniform): no white-noise stretch, quality similar to the ML control, and WB vs KN indistinguishable by eye at this depth.
5. The mlp sample is texturally indistinguishable from the bigram table sample: matched val NLL implies matched sample texture here, closing the loop on learning #1 in the generation regime.
6. No rung produces grammatical English beyond a short phrase; even KN at depth 4 is locally Shakespearean and globally word salad.

**Result.**
Recomputed val NLLs match the ladder and scoreboard: unigram 3.3276, bigram 2.5085, trigram 2.1320, 5-gram (a=0.01) 1.8755, WB depth 4 1.7898, KN depth 4 1.6539, mlp 2.5094.
Primer is `\n\nISABEL`; the whole 5-gram family completes it to "ISABELLA:" and, sharing seed and near-identical top distributions, opens with the same "LA:\nTruly" before diverging.
Samples (seed 0, first ~200-280 of 400 chars shown; `pixi run sample` reproduces the full output):

random (uniform), val NLL 4.174:

```text
qcUEH!& ;nduTayicWXvFoe
MqX!kiq;'r W'GSON
.
eVdDbkLQznyLfwdpfgMr.YhpVLHOShs&vVKeYDHhZTIkMIsEBhb$'LpNmHCmr'$eIY3qQtmgBk$YNz?w'bYtGtes?kw$Kc,UbkvNPRx?T$OwbJzaw QpkNTOVBm'NFijhvv?-.iyvexr
```

unigram table (a=0.01), val NLL 3.328:

```text
lI
rtiohur
s
o shTe
 mljcywnlnd ogWettahaiadt:l srAt a er: d, iSm,ub lteueekyueoehrdont otw
sxu wtre;rtIhet oi
o
oft s aewhGAt: OihrhOdrlvbhis dt rds
b mKnu s ceewpWIstfaya trmvtosC mn dthi'gg yh
```

bigram table (a=0.01), val NLL 2.5085:

```text
OF ETornetl
UChathan ashe:'thend se!
Wine, ges o tilo g is LI ar?
As,
Nonthelyoounerdent stt two? tle tr, it sh s
Tht t gethalof finor:
Thuliou, vater h pent t hedve!
Vindy busitroue rl!
Whe me y,
Turais nfu mod Pl hooust
```

trigram table (a=0.01), val NLL 2.132:

```text
LA:
To sour
gar; he Pater,--and and my, me;
Fore to the loo in peo, fartay istripent?
At tor try woul.
Senclask taret the wilet all the thy mot it thich reiver heave froseve to wour bromerech, OxfEly gived wast ife ithous a the reir; I conumsed:
Romfory, holl.

QUEEN EDWARETHUMNIUS:
What;
```

5-gram table (a=0.01), val NLL 1.8755 - the best fixed table by NLL:

```text
LA:
Truly;V
q!i;rWGO .edbLzyfdfM.hVHSsvKYHZIMsBb'pmCr$I3QmB$N?'YGe?wK,bvPxTObzwQkTVmNihv-ivx
ryx3ysoSCnvEWPv!ia h!kUv&p&JOxXDCsB.FZXnXFNobxLWZq3Nu!oNo
K'dEgw.r$LOSylHErsUJzH;snexvjqD3eh:MuXY?VV'yY
```

5-gram table (a=0, ML control), val NLL inf:

```text
LA:
Truly:' mindeed,
And know, gentlement of the stay hand Quintus wife:
No.

VOLUMNIA:
Away withough to men
That thy see you that my goodly prayers take hit the lusterously to-night so hope of yonder follow not an exerciful.

CORIOLANUS:
Your spot, here it spirit! mulberdsmen worst not pardon of thou should before, as long and weeders,
```

Witten-Bell backoff, depth 4, val NLL 1.7898:

```text
LA:
Truly:'
PAULINA:
Even your gentlement of the stay happier's forth such divorce. I'll susestrawing to men
That thy see you that my goodly prayers take hit the lust. What but with all him,
If yondition of.

QUEEN ELIZABETH:
Nay, her mastiny:
Whose here it spirit!

COMINIUS:
It is, them all in arms:
```

Kneser-Ney backoff, depth 4, val NLL 1.6539:

```text
LA:
Truly:'
Passes i' the your gentlement of the stay eithee can set any whereof, or soul: wertainst myself;
So say you have does rest my wish:
Stay war these and you to hundred, stampeditch left His good number,
That with and bring the six years, I have knees,
That al robest:
Withink, so have offering not,--
```

mlp (per-position, e32 h128), val NLL 2.5094:

```text
is remhis keswhenlteredoca meaman youd INAlste grecofuesy hifoty here hath thig berelisseanongilost atho y we w ke e alces idy t, tr nd-d han tsoo we w y mer aim whioervoto,
Preschan s uzed
I s h. fo INVeprf pifie-mily m,-ennd ssay eizecodeld fleome inou inkee he t,
Ase.
```

Prediction 1 confirmed, with one underestimate: the corpus space fraction is 15.2% (predicted "roughly 16%"), the ladder of textures is exactly as described, but the full NAME-colon-newline speaker format (blank line, capitalized blend-name, colon) is already fully formed at the trigram ("QUEEN EDWARETHUMNIUS:", "KINA:"), not merely hinted at.
Prediction 2 confirmed, more violently than predicted: the a=0.01 5-gram escapes at generated char ~11 ("Truly;" then "V") and never recovers, so ~97% of the sample is uniform white noise, visually indistinguishable from the random model, despite the best fixed-table val NLL.
Prediction 3 confirmed: the a=0 control stays Shakespeare-shaped for all 400 chars (VOLUMNIA, CORIOLANUS, and the blend "DUKE OF GAUNT" further in) with infinite val NLL.
Prediction 4 confirmed: WB and KN never derail, and at this depth neither WB vs KN nor either vs the ML control is rankable by eye; the first divergence point is exactly where smoothing policy differs (a=0.01 samples ";V" off-table where ML/WB/KN sample ":'").
Prediction 5 confirmed: the mlp (2.5094) and the bigram table (2.5085) produce the same digraph soup with accidental short real words ("here hath", "youd", vs "Wine", "me by"); matched NLL came with matched texture.
Prediction 6 confirmed: everything is word salad above the phrase level; KN's longest locally coherent run is on the order of "That with and bring the six years, I have knees".

**Conclusion / next steps.**
Val NLL and generation quality dissociate in both directions at the same context depth: the best fixed-table NLL (1.876) produced white noise, and infinite val NLL (a=0) produced one of the best-looking samples.
The mechanism is the regime, not the model class: NLL scores one-step prediction on gold histories, while generation compounds sampling errors; the fixed table's only escape hatch from a low-count context is smoothing mass, which teleports to uniform over an absorbing set of novel contexts, whereas backoff degrades one context length at a time and always returns.
Smoothing policy, worth only 0.086 nats of NLL (a=0.01 1.876 vs WB 1.790), is the difference between noise and Shakespeare-shaped text once the model eats its own output.
Texture otherwise tracks information access, confirming learning #1 in the generation regime: the 15k-param mlp writes exactly like the 65x65 count table it is information-equivalent to, down to a 0.001-nat NLL match.
Next: sample the context-aware neural models (rnn / lstm / attention at context 32) against KN - do their 1.9-2.0 NLLs read better or worse than KN's 1.65 (word inventory vs longer-range structure)? And check whether the a=0.01 collapse time scales as predicted (~1 / (0.63 K / N) chars) at other depths.

## 2026-07-14: The params-vs-loss frontier per architecture (3.9k / 15k / 60k, fixed budget)

**Question.**
The tiny-budget rematch left two constraints compounded: all four context-aware models cluster 0.12-0.26 nats below the bigram floor, and the seed-0 curves are still falling 0.07-0.10 nats over the final 2000 iters (checked from curves/*.csv), so the cluster is a budget-capped snapshot, not an asymptote.
This entry isolates the parameter axis at the same training budget: a 3.9k / 15k / 60k frontier per architecture, to locate where each mechanism's capacity stops binding, where concat re-catches the rnn (the 15k-era tie broke at 3.9k), and whether width lifts attention off the bottom of the context-aware group.

**Method.**
Same budget and setting as the tiny rematch: batch 32 x seq_len 8 (256 tokens/step), lr 1e-3, 5000 iters = 1.28M tokens, 3 seeds, context_k 8 for the window models, appended to runs.jsonl.
The 3.9k tier reuses the tiny-rematch runs unchanged (identical config).
New tiers at ~15k and ~60k (+/-4%), dims found by closed-form param search holding each architecture's 3.9k aspect ratio (H/E) fixed, so the frontier varies capacity, not shape:

| model | 15k config | 15k params | 60k config | 60k params |
|---|---|---|---|---|
| mlp | E50 H100 | 15,315 | E130 H260 | 60,515 |
| mlp_sum | E36 H120 | 14,645 | E96 H320 | 58,145 |
| mlp_concat | E20 H60 | 14,925 | E44 H132 | 58,101 |
| rnn | E42 H68 | 14,763 | E102 H166 | 62,139 |
| lstm | E38 H34 | 14,673 | E86 H76 | 60,147 |
| attention (1L, 4h) | E28 FFN144 | 15,497 | E60 FFN320 | 61,945 |

**Prediction (pre-registered, by the agent).**
1. mlp is the flat control: 2.51-2.54 at every tier, total 3.9k -> 60k gain <= 0.02 (information-bound per learning #1), dissociating information-bound flatness from capacity-bound gains everywhere else.
2. Every other architecture gains >= 0.10 from 3.9k -> 60k, and attention gains the most (>= 0.25): capacity is first-order for every context-aware mechanism at this size, and width is what attention lacks (learning #12 downscaled).
3. concat re-catches the rnn gradually but does not pass it: the rnn-concat gap shrinks monotonically, 0.050 (3.9k) -> <= 0.03 (15k) -> <= 0.015 (60k), rnn ahead or tied at every tier (weight sharing loses decisiveness as E grows past the per-offset cost; the old 31k/43k pair was already a statistical tie).
4. 15k bands: rnn 2.10-2.17, concat 2.13-2.20, lstm 2.15-2.25, attention 2.15-2.22; tiny ordering preserved (rnn < concat < lstm < attention).
5. 60k bands: rnn 1.99-2.06, attention 2.02-2.10, concat 2.04-2.10, lstm 2.04-2.12.
6. Riskiest call - exactly one ordering inversion, at the top tier: attention passes lstm at 60k (width feeds attention faster than the gate tax relents on a squeezed state), giving rnn < concat < attention < lstm at 60k.
7. mlp_sum stays strictly worst at every tier but gains >= 0.10 by 60k: at E96 > vocab 65 the embeddings can be linearly independent, so the sum can encode the bag losslessly and superposition interference (the mlp_sum entry's optimization diagnosis) should mostly dissolve; the residual gap is recency dilution, which capacity cannot buy back. Sharp version: sum at 60k still at or above the bigram floor (>= 2.50).
8. Nobody approaches the context-8 information floor, and the frontier visibly bends: best model at 60k stays >= 0.15 above the position-averaged KN floor of ~1.83 (mean of KN val at k=1..8), and per architecture the 15k -> 60k gain is smaller than the 3.9k -> 15k gain.

**Result.**
Val loss, mean +/- std over 3 seeds (54 runs total; 3.9k column = tiny-rematch runs):

| model | 3.9k | 15k | 60k | total gain |
|---|---|---|---|---|
| mlp | 2.524 +/- 0.005 | 2.512 +/- 0.007 | 2.515 +/- 0.007 | 0.010 |
| mlp_sum | 2.822 +/- 0.018 | 2.676 +/- 0.010 | 2.617 +/- 0.005 | 0.205 |
| mlp_concat | 2.315 +/- 0.007 | 2.148 +/- 0.007 | 2.050 +/- 0.007 | 0.265 |
| rnn | 2.265 +/- 0.018 | 2.094 +/- 0.008 | **1.991 +/- 0.013** | 0.274 |
| lstm | 2.358 +/- 0.009 | 2.162 +/- 0.005 | 2.029 +/- 0.002 | 0.329 |
| attention | 2.400 +/- 0.008 | 2.186 +/- 0.010 | 2.085 +/- 0.016 | 0.316 |

Orderings: 3.9k and 15k both give rnn < concat < lstm < attention < mlp < sum; 60k gives rnn < **lstm < concat** < attention < mlp < sum.
The mlp tier curves (all three capacities collapsing onto the bigram floor, larger models arriving in fewer tokens but no lower) are plotted in curves/mlp_capacity_nll_vs_tokens.png; even on the train split the 60k mlp stops at 2.468, just above the train-split bigram table floor of 2.452 - structurally blind models cannot even memorize past their information floor.
Prediction 1 confirmed: the mlp frontier is flat to 0.010 nats across 15.7x parameters (and non-monotone 15k -> 60k within noise).
Prediction 2 half-confirmed: every context-aware architecture gains 0.205-0.329, but the largest gain belongs to the lstm (0.329), not attention (0.316), by a ~1-sigma margin.
Prediction 3 REFUTED in the interesting direction: the rnn-concat gap does not shrink - it is scale-stable at 0.050 / 0.054 / 0.059, rnn ahead at every tier.
Concat never re-catches the rnn at matched parameters; the old "tie" (2.070 vs 2.088) compared a 31k rnn against a 43k concat.
Prediction 4 confirmed (ordering exact; 3 of 4 bands hit, rnn 0.006 better than its band).
Prediction 5 mostly confirmed (3 of 4 bands; lstm 0.011 *better* than its band).
Prediction 6 (riskiest) REFUTED: exactly one inversion did occur at 60k, but it was the lstm passing concat (2.029 vs 2.050, ~3 sigma), not attention passing the lstm; attention stays last among context-aware models at every tier.
Prediction 7 confirmed in all three clauses: sum is strictly worst everywhere, gains 0.205, and still sits at 2.617 >= 2.50 at 60k - 0.10 nats *above* the no-context mlp even though E96 >= vocab makes the bag losslessly encodable.
Prediction 8 confirmed: rnn at 60k (1.991) sits 0.161 above the ~1.83 position-averaged KN floor, and every architecture's 15k -> 60k gain is smaller than its 3.9k -> 15k gain (e.g. attention 0.214 -> 0.102, rnn 0.171 -> 0.103).
Train/val gaps grow with capacity (0.02-0.05 at 3.9k, 0.06-0.10 at 15k, 0.11-0.16 at 60k) but no cell shows a val upturn: capacity is beginning to outrun 1.28M tokens without yet overfitting.

**Conclusion / next steps.**
The information-bound vs capacity-bound dissociation is now measured, not argued: 15.7x parameters moves the context-blind mlp 0.010 nats while every context-aware mechanism gains 0.21-0.33.
The tiny-rematch's headline finding reverses at scale in one place only: weight sharing's decisive advantage (rnn over concat) is *not* a scarcity artifact - the 0.05 gap survives 3.9k -> 60k unchanged - but the lstm's gate tax is: once H clears ~76, gating overtakes fixed per-offset weights, and the recurrent family occupies the top two slots at 60k.
Attention's reputation is only half-rehabilitated by capacity: width gives it the biggest single jump at the first quadrupling, yet it remains the worst context-aware mechanism at every tier - at context 8 and 1.28M tokens, width is necessary but not sufficient.
The sum model's deficit is informational, not optimizational: with a lossless bag encoding available it still loses to no-context, evidence that H(next | bag-8) genuinely exceeds H(next | last char) - recency dilution is a property of the *representation*, and no capacity buys it back.
Everything bends: returns per 4x params are decelerating at fixed budget, and the whole neural family is still >= 0.16 above the classical floor, with train/val gaps warning that the next quadrupling starts paying rent to overfitting.
Next: a 240k tier (does the lstm's steeper frontier cross the rnn's, as extrapolation suggests?); the same frontier at 4x iters to unconfound the still-falling curves; and the seq_len 32 six-way comparison still standing from the tiny rematch.

## 2026-07-14: The 15k capacity-matched context ladder - concat squeezed to the twin budget

**Question.**
The 141.6k controls put the fixed-capacity context optimum near k=8 and showed parameters pay only where the window supplies unextracted information.
Now squeeze the same ladder down to the reference budget: hold E=32 and match mlp_concat to the k=1 twin's 14,689 params by shrinking H as k grows.
Where does the context optimum move at 15k params, and how steep does the deep tail get when every added offset is paid for out of a fixed budget?

**Method.**
Same base as the main grid (seq_len 32, batch 32, lr 1e-3, 5000 iters, 3 seeds); mlp_concat with H solved from params = 2,145 + H(32k + 66) ~= 14,689.
Cells: k=2 H=96 (14,625), k=3 H=77 (14,619), k=4 H=65 (14,755), k=5 H=56 (14,801), k=8 H=39 (14,703), k=16 H=22 (14,861), k=32 H=12 (15,225).
All within 1.2% of target except k=32 at +3.7% (inside the rematch's +/-4% convention; the excess favors the cell predicted to lose, so the bias is conservative).
k=1 is the existing twin cell (H=128, 14,689), not rerun.
Because sum is 14,689 at every k, this grid also makes concat vs sum an exact param-matched mechanism comparison at every depth.

**Prediction (pre-registered, by the agent).**
1. The U sharpens and its minimum shifts left as capacity shrinks: minimum at k in {3, 4, 5} with value in [1.90, 1.97], and k=8 worse than that minimum by >= 0.03 with separated seed ranges (at 141.6k params the optimum was k=8).
2. Every squeezed cell lands above its H=128 counterpart with a penalty monotone in k: within +0.03 at k=2 (capacity was worth ~nothing there), +0.06 to +0.15 at k=8, and +0.20 to +0.50 at k=32 (H=12) - the tail tax is paid in share-of-budget, so it steepens as the budget shrinks.
3. Mechanism still beats bag at exactly matched params: concat below sum at every k >= 2, but the k=32 margin narrows from 0.98 (H=128 concat) to [0.4, 0.8].
4. No cell beats 1.90: at this budget capacity binds before mechanism does, and the 141.6k ladder's 1.833 stays out of reach.

**Result.**
Train / val (nats/char), val as mean +/- std over 3 seeds; 21 runs appended to runs.jsonl; all three concat ladders plus the mlp_sum ladder plotted in curves/context_ladders.png.

| k | H | params | train / val | gap |
|---|---|---|---|---|
| 1 | 128 | 14,689 | 2.457 / 2.505 +/- .004 | 0.047 |
| 2 | 96 | 14,625 | 1.990 / 2.124 +/- .002 | 0.134 |
| 3 | 77 | 14,619 | 1.833 / **1.993** +/- .008 | 0.160 |
| 4 | 65 | 14,755 | 1.822 / 1.994 +/- .009 | 0.172 |
| 5 | 56 | 14,801 | 1.838 / 2.002 +/- .007 | 0.164 |
| 8 | 39 | 14,703 | 1.894 / 2.045 +/- .008 | 0.151 |
| 16 | 22 | 14,861 | 2.012 / 2.135 +/- .004 | 0.123 |
| 32 | 12 | 15,225 | 2.152 / 2.240 +/- .006 | 0.088 |

Prediction 1 confirmed on location, refuted on level: the minimum is at k=3 (1.993, statistically tied with k=4 at 1.994), inside the predicted {3, 4, 5}, and k=8 is worse by 0.052 with separated seed ranges - but 1.993 sits 0.02 above the predicted [1.90, 1.97] band.
Prediction 2 confirmed in full: the penalty vs the H=128 counterpart is monotone in k - +0.010 / +0.042 / +0.080 / +0.101 / +0.130 / +0.203 / +0.286 at k=2 through 32 - with every checkpoint clause inside its band.
Prediction 3 confirmed: concat stays below sum at every k >= 2 at exactly matched params, margin 0.199 at k=2 widening to 0.691 at k=32, inside [0.4, 0.8].
Prediction 4 confirmed: best cell 1.993; the 141.6k ladder's 1.833 is out of reach at this budget.
Unpredicted finding: the 15k tail rises for the *opposite reason* from the H=128 tail.
At H=128 the tail was a variance tax (train flat at ~1.70, gap growing 0.195 -> 0.246); at 15k the gap *shrinks* along the tail (0.172 -> 0.088) while train *rises* (1.822 -> 2.152) - H=12 can no longer even fit the training data.
Same U shape, opposite anatomy: capacity-rich tails fit noise, capacity-starved tails stop fitting signal.

**Conclusion / next steps.**
Optimal context scales with capacity: 14.7k params -> k=3, the H=128 diagonal -> k=5, 141.6k -> k=8 - a monotone optimum-k(params) relation measured at three budgets.
The val-only U-curve is degenerate evidence: two mechanically opposite tail failures (fit noise vs cannot fit signal) produce the same shape, and only the train column dissociates them - read train before explaining val.
Mechanism survives the squeeze: at exactly equal params, ordered per-offset weights beat the bag at every depth, so the 0.2-nat order premium at k=2 is budget-robust and grows with depth.
Next: a third and fourth budget (~40k, 60k) to test whether optimum-k(params) follows a power law; weight decay on the capacity-rich k=32/H=128 cell to test whether its variance-tax tail flattens (the capacity-starved tail should not respond).

## 2026-07-14: Context ladder at fixed capacity - mlp_sum vs mlp_concat, k = 1 to 32

**Question.**
The classical ladder showed three deep-context signatures: fixed tables U-turn, WB saturates, KN's tail is free.
Fix neural capacity at E32/H128 and walk context_k up the same depths (1, 2, 3, 4, 5, 8, 16, 32): which signature does each window mechanism follow - the order-blind bag (mlp_sum) and the ordered window (mlp_concat)?

**Method.**
`sweep` over the model x context_k grid at BASE_CONFIG + dim_embed 32, dim_hidden 128, seq_len 32; batch 32, lr 1e-3, 5000 iters, 3 seeds, appended to runs.jsonl.
Supervision is 1024 tokens/step (5.12M tokens ~ 5.7 epochs), 8x the old 128-token standard setting, so cross-entry comparisons carry the learning-#10 caveat.
Loss is position-averaged over all 32 positions (warm-up caveat, learning #9), unlike the classical ladder's i >= 32 scoring, so classical-column comparisons are directional only.
Params: sum is constant at 14,689 for every k; concat is 10,593 + 4,096k (14,689 at k=1 up to 141,665 at k=32) - for concat the k axis is confounded with params by construction, since per-offset weights *are* the mechanism.
Control arm (queued after the main grid): capacity-matched concat cells that isolate params-without-context by raising H at small k to match k=32's 141,665 params - k=2 at H=1073 (141,635) and k=8 at H=433 (141,571), both within 0.1% of target, 3 seeds each.
First sweep through the new model-saving path: every run's final weights land in models/ with model_path recorded in its runs.jsonl row.

**Prediction (pre-registered, by the agent).**
1. k=1 twins: mlp_sum and mlp_concat at k=1 build identical architectures (same tensor shapes in the same RNG draw order, same batch stream), so per-seed losses agree to float precision - the strictest seed-discipline check yet.
Both land in 2.48-2.52 (the neural bigram at this supervision).
2. mlp_sum traces a U-curve like the fixed table but for a different reason (dilution, not sparsity): best at k=2 (2.38-2.46, edging the 128-token cell 2.462), k=3 within +0.06 of k=2, k=4 back above the k=1 bigram line (the learning-#4 boundary), then monotone worse - k=8 in [2.60, 2.80], k=16 in [2.70, 3.00], k=32 in [2.80, 3.20], while staying below unigram 3.328 (a bag of 32 still beats no context).
3. mlp_concat follows the KN signature: monotone improvement with diminishing returns and a free tail, no U-turn.
Bands: k=2 [2.18, 2.30], k=3 [2.02, 2.15], k=4 [1.93, 2.06], k=5 [1.87, 2.00], k=8 [1.78, 1.92], k=16 [1.74, 1.88], and k=32 within 0.04 of its own k=16 value.
Generalization is what makes deep context free for a parametric model - the property KN has to buy with discounting.
4. The order gap (sum minus concat at equal k) widens monotonically: ~0 at k=1, 0.15-0.25 at k=2 (learning #3's ~0.2 nats), above 0.8 nats by k=32.
5. Memorization column: concat k=32 (141k params, ~5.7 epochs) shows the grid's largest train/val gap, 0.08-0.20; sum's gap stays under 0.05 at every k (14.7k params cannot memorize 892k chars).
6. No cell beats 1.75: at 5.12M tokens supervision is no longer the binding constraint, the one-layer window mechanism is, and KN's 1.634 stands.
7. Capacity cannot substitute for context: the H-matched k=2 control (7.5x its H=128 cell's params) gains only 0.05-0.15 nats and stays above the H=128 k=5 cell; the H-matched k=8 control (3.3x params) gains 0.03-0.10 and stays above the H=128 k=16 cell.
Sharpest form: the capacity share of the k=2 -> k=32 concat improvement is under one third.

**Result.**
Train / val (nats/char), val as mean +/- std over 3 seeds; 48 + 6 runs appended to runs.jsonl, weights in models/.

| k | concat params | concat train / val | sum train / val (14,689 params) |
|---|---|---|---|
| 1 | 14,689 | 2.457 / 2.505 +/- .004 | bit-identical to concat |
| 2 | 18,785 | 1.978 / 2.114 +/- .002 | 2.206 / **2.323** +/- .002 |
| 3 | 22,881 | 1.770 / 1.951 +/- .005 | 2.224 / 2.383 +/- .006 |
| 4 | 26,977 | 1.722 / 1.914 +/- .007 | 2.352 / 2.500 +/- .006 |
| 5 | 31,073 | 1.707 / **1.901** +/- .008 | 2.488 / 2.629 +/- .006 |
| 8 | 43,361 | 1.696 / 1.915 +/- .009 | 2.660 / 2.776 +/- .010 |
| 16 | 76,129 | 1.699 / 1.932 +/- .010 | 2.819 / 2.891 +/- .006 |
| 32 | 141,665 | 1.708 / 1.954 +/- .011 | 2.878 / 2.931 +/- .005 |

Controls: concat k=2 H=1073 (141,635 params) 1.956 / 2.117 +/- .005; concat k=8 H=433 (141,571 params) 1.547 / **1.833** +/- .005.

Prediction 1 confirmed in its strictest form: per-seed val curves are bit-identical between the k=1 twins (2.5050 / 2.4994 / 2.5092), and the mean sits in the predicted band.
Prediction 2 confirmed in shape, refuted in details: sum's minimum is at k=2 but at 2.323, 0.06 *below* the predicted floor (supervision undervalued again), and the learning-#4 boundary moved - k=4 now ties the bigram line (2.500 vs 2.505) instead of exceeding it, pushing the worse-than-no-context crossover to k=5; the deep cells landed in-band and below unigram.
Prediction 3 half-refuted, and the refuted half is the finding: k=2-4 beat their bands, but the tail is *not* free - concat U-turns at k=5 and rises to 1.954 by k=32 with cleanly separated seed ranges (the within-0.04-of-k=16 clause survived at +0.022; the no-U-turn clause did not).
The train column gives the mechanism: train is flat at ~1.70 from k=4 on, so offsets past ~5 extract nothing new, while the train/val gap grows 0.195 -> 0.246 - the val rise (+0.052) is the gap rise (+0.051), a pure variance tax.
Prediction 4 confirmed: order gap 0.000 / 0.208 / 0.978 at k=1/2/32 - the third independent ~0.2-nat measurement of order in a 2-char window.
Prediction 5 half-refuted: concat k=32 does carry the grid's largest H=128 gap but at 0.246 it tops the predicted range, and sum's no-memorization clause is wrong - constant 14.7k params show gaps up to 0.159 (k=3), non-monotone in k, so the gap is partly split shift interacting with the window, not capacity memorization.
Prediction 6 confirmed: best H=128 cell 1.901, best overall cell 1.833, both above 1.75; KN's 1.634 stands.
Prediction 7 split, and the split is the second finding: at k=2, 7.5x params bought -0.003 (predicted 0.05-0.15 - refuted *low*: capacity alone is worth zero, train barely moved 1.978 -> 1.956), while at k=8, 3.3x params bought 0.082 (in-band; train 1.696 -> 1.547) and the control beat every H=128 cell including the k=5 minimum, refuting the stays-above-k=16 clause.
The sharpest form (capacity share under one third) is confirmed at the k=2 anchor, but the true structure is an interaction: parameters pay only where the window supplies unextracted information.

**Conclusion / next steps.**
Neither neural window model earns KN's free tail: sum U-turns from dilution, concat U-turns from a parameter-variance tax - but generalization softens the deep-context penalty ~40x relative to counting (fixed table +2.30 from its minimum to k=32, concat +0.05).
The tail's cost tracks what depth *buys*: free when depth costs no parameters (KN via discounting; rnn/lstm via recurrence per the earlier grids), taxed when every extra step buys weights (concat) and poisonous when it dilutes the signal (sum).
The three 141.6k-param cells form the capacity-matched context ladder the confound question demanded: k=2 2.117, k=8 **1.833**, k=32 1.954 - at fixed parameters the context optimum is ~8, and the H=128 grid's k=5 minimum was partly an artifact of starving the deeper cells of width.
Learning-#14's dissociation now reproduces *within* one architecture along its own context axis: at k=2 the model is information-bound (7.5x params buys 0.000), at k=8 it was capacity-bound (3.3x buys 0.082).
Cross-entry comparisons carry the 8x-supervision caveat: concat k=8 H=433's 1.833 is not scoreboard-comparable to the 128-token cells, and KN still leads by 0.20 on a shared-split basis.
Next: an H sweep at k=8 to find where width saturates (gap 0.286 says regularization or early stopping may pay before more width does); weight decay on the k=32 cell to test whether concat's tail tax is removable - if regularization makes the tail free, the mechanism story sharpens to "the U-turn is pure estimation noise, not representation".

## 2026-07-14: All six architectures at ~3.9k params and 1.28M tokens (the tiny-budget rematch)

**Question.**
The new BASE_CONFIG shrinks the reference mlp to 3,857 params and raises supervision to 256 tokens/step (batch 32 x seq_len 8; 5000 iters = 1.28M tokens ~= 1.4 epochs).
Where do the six neural architectures land when *both* parameters (~3.9k, +/-4%) and training tokens (1.28M) are matched, at context <= 8?
Prior comparisons matched neither: the old grid had params ranging 15k-31k at 128 tokens/step.

**Method.**
Param-matched configs found by dim search against the mlp reference (vocab 65, seq_len 8, context_k 8 for the window models):
mlp E16/H32 (3,857), mlp_sum E12/H40 (3,965), mlp_concat E8/H24 (3,705), rnn E16/H26 (3,913), lstm E16/H14 (3,751), attention 1L/4h E12/FFN64 (3,993).
All at batch 32, seq_len 8, lr 1e-3, 5000 iters, 3 seeds, appended to runs.jsonl; seed-0 curves exported to curves/*.csv.
Same seed => bit-identical batch sequences across models (batches come from np.random, init from torch).

**Prediction (pre-registered, by the agent).**
1. mlp stays pinned at the bigram floor: 2.50-2.53 val (seed-0 run already showed 2.518).
2. mlp_sum k=8 remains worse than no context at all: 2.70-2.85 (bag dilution + superposition is representational, so the capacity cut changes little).
3. concat and rnn stay within 0.05 of each other (context 8 saturates both mechanisms per the architecture-vs-context entry) and land in 2.05-2.20: the 4x param cut costs a little, the 2x supervision gives a little back.
4. lstm loses to rnn by >= 0.03: at 3.8k params the gate tax squeezes state width to 14 vs the rnn's 26, and state width binds at equal params (learning #11).
5. attention is the worst context-aware model: 2.25-2.40 at E12 (it was 2.218 at E32/17.9k params, and width is what attention needs most).
6. Full predicted val ordering: rnn ~= concat < lstm < attention < mlp < sum.

**Result.**
Val loss, mean +/- std over 3 seeds (train/val curves for seed 0 in curves/*.csv):

| model | params | dims | val loss |
|---|---|---|---|
| rnn | 3,913 | E16 H26 | **2.265 +/- 0.018** |
| mlp_concat | 3,705 | E8 H24 k8 | 2.315 +/- 0.007 |
| lstm | 3,751 | E16 H14 | 2.358 +/- 0.009 |
| attention | 3,993 | E12 FFN64 1L | 2.400 +/- 0.008 |
| mlp | 3,857 | E16 H32 | 2.524 +/- 0.005 |
| mlp_sum | 3,965 | E12 H40 k8 | 2.822 +/- 0.018 |

Predictions 1, 2, 4, 5 confirmed: mlp pinned at the bigram floor (2.524); sum worse than no context (2.822); lstm loses to rnn by 0.093 (gate tax at H=14 vs H=26); attention worst context-aware model (2.400, at the top of its predicted band).
Prediction 3 half-refuted: the level band was wrong (both landed 0.10-0.11 above 2.20 - the 4x param cut cost more than the 2x supervision returned), and the "concat ~= rnn" tie from the 15k-param grid did *not* survive the squeeze: rnn wins by 0.050 with non-overlapping seed ranges.
Prediction 6 (full ordering) confirmed exactly: rnn < concat < lstm < attention < mlp < sum.
Train/val gaps are 0.02-0.04 everywhere: capacity/budget-bound, not overfitting.

**Conclusion / next steps.**
The mechanism ladder is scale-robust downward: the same ordering as the 15k-31k grid holds at 3.9k params.
But the earlier "mechanism is invisible at short context" finding is a *capacity-conditional* fact, not a universal one: when parameters are scarce, weight sharing becomes the decisive mechanism property - concat must buy each offset its own weights out of an E=8 embedding budget, while the rnn reuses one cell, and the tie breaks 0.05 in the rnn's favor at the same context.
Gating (lstm) and dynamic selection (attention) are luxuries the budget cannot pay for at 3.9k params; both invert their large-scale reputations.
Every model sits 0.59+ nats above the KN depth-8 floor (1.659): tiny neural models are nowhere near counting on this data.
Next: the same six-way comparison at seq_len 32 to see whether the rnn's win grows with context reach, and a params-vs-loss frontier per architecture (3.9k / 15k / 60k) to locate where concat re-catches the rnn.

## 2026-07-14: The full classical ladder to k=32 - train columns and the deep-context tail

**Question.**
The classical models have never been run past context 8 (KN was "capped at depth 8; deeper levels are memory-prohibitive"), and the backoff methods have no train-side numbers at all.
What do train and val NLL look like for the fixed table (alpha=0.01), Witten-Bell, and Kneser-Ney at k in {1, 2, 3, 4, 5, 8, 16, 32} - in particular, do the backoff ladders keep rising past k=8, and what does the train/val gap say about how much each method memorizes?

**Method.**
New `deep_ladder` in `experiments/ngram.py`: instead of materializing all k+1 backoff levels at once (the old memory cap), stream one count level at a time and fold it into a running per-position interpolated probability - WB applies every level in sequence, and KN maintains its lower-order continuation chain q with the raw top level applied at each requested depth.
Peak memory becomes ~2 levels regardless of depth; tables are bytes-keyed.
All methods are scored on positions i >= 32 of each split so every cell shares one prediction set (earlier entries scored i >= k; shifts numbers by < 0.005, checked against the known k <= 7 values).

**Prediction (pre-registered).**
1. Fixed table (a=0.01): val reproduces the known U-curve (min 1.876 at k=4, ln 65 = 4.174 by k=32) while train falls monotonically to ~0.49 at k=32 - the pure-memorization floor, since nearly all 32-char train contexts are singletons and -ln(1.01/1.65) = 0.49.
2. WB val keeps rising past its k=4 minimum but saturates: k=16 lands in [2.0, 2.3] and k=32 is within 0.05 of k=16, because an unseen long context is skipped by the recursion, and val positions whose deep contexts do match train become rare quickly with depth.
3. KN val stays in [1.63, 1.69] at every k >= 5: no U-turn, deep context is harmless under discounting, and no new best (1.634 at k=5 stands within noise).
4. The train/val gap at k >= 8 orders the methods by memorization: fixed >> WB >> KN. Sharpest form: WB train falls below 0.1 by k=32 (once train contexts are unique, every level is a singleton with lam = 0.5 and ML = 1, so the chain pushes p to 1 geometrically), while KN train stays above 0.2 because discounting caps what a singleton can contribute.

**Result.**
Train / val NLL (nats/char), all cells scored on positions i >= 32 of each split; known k <= 7 values reproduced to ~0.001 first.

| k | contexts | fixed a=0.01 | WB backoff | Kneser-Ney |
|---|---|---|---|---|
| 1 | 65 | 2.449 / 2.509 | 2.449 / 2.504 | 2.449 / 2.504 |
| 2 | 1,360 | 1.899 / 2.132 | 1.901 / 2.105 | 1.900 / 2.100 |
| 3 | 10,899 | 1.491 / 1.885 | 1.497 / 1.866 | 1.496 / 1.796 |
| 4 | 46,224 | 1.235 / **1.876** | 1.236 / **1.790** | 1.243 / 1.654 |
| 5 | 125,229 | 1.045 / 2.118 | 1.023 / 1.828 | 1.051 / **1.634** |
| 8 | 508,736 | 0.657 / 3.253 | 0.455 / 2.103 | 0.632 / 1.659 |
| 16 | 864,915 | 0.493 / 4.142 | 0.023 / 2.326 | 0.431 / 1.658 |
| 32 | 891,327 | 0.491 / 4.174 | 0.0002 / 2.334 | 0.422 / 1.658 |

Prediction 1 confirmed to the third decimal: fixed-table train bottoms out at 0.4905 vs the predicted singleton floor -ln(1.01/1.65) = 0.4914, and val hits ln 65 = 4.1744 exactly at k=32.
Prediction 2 split: the saturation clause is confirmed (k=32 is within 0.008 of k=16) but the level band is narrowly refuted, k=16 = 2.326 vs the predicted [2.0, 2.3] - the k=8 -> 16 rise (+0.22) was underestimated, since levels 9-16 still have 618k-865k distinct contexts and therefore still fire often on val.
Prediction 3 confirmed: KN sits at 1.634 / 1.659 / 1.658 / 1.658 from k=5 on - no U-turn, and the deep tail is even epsilon-favorable (k=16 and 32 land a hair below k=8).
The depth-8 cap in the attention-comparison entry is retroactively validated: exact KN at k=16/32 (1.658) vs the capped value quoted there (1.659).
Prediction 4 confirmed in its sharpest form: at k=32 the train/val gaps are fixed 3.68 >> WB 2.33 >> KN 1.24, and WB train is 0.0002 - the WB chain reproduces the training text essentially byte-for-byte, exactly the geometric singleton argument (every unique-context level blends in ML = 1 at lam = 0.5).

**Conclusion / next steps.**
The three classical methods now have complete memorization signatures, and they cleanly dissociate memorization from val behavior.
The fixed table memorizes (train 0.49) and collapses (val 4.17).
WB memorizes *harder* (train 0.0002 at k=32, a near-lossless copy of the corpus) yet only saturates on val: its singleton trust means every deep level is pure memorization, which costs it 0.54 nats of val vs its own k=4 best because matched-but-divergent val contexts get their probability halved level after level.
KN refuses most singleton evidence (train stays at 0.42) and in exchange is the only method whose deep-context tail is free: val flat at 1.658 from k=8 to k=32.
So discounting is not just a fix for the U-turn - it is what makes context extension costless, the property the neural models get from generalization.
The classical scoreboard is unchanged: KN 1.634 at k=5 remains the number to beat.
The streaming evaluator (`deep_ladder`) removes the old depth-8 memory cap for future deep-context comparisons.
Still standing: modified KN (per-count discounts) for a tighter floor; refreshing the lstm+KN mixture with the 80k-iter lstm.

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
Final val loss by budget (5k = 3-seed mean; 20k/80k = seed 0; eval noise ~+/-0.01-0.02):

| model | params | 5k | 20k | 80k final | 80k min (@ iter) | train @ 80k |
|---|---|---|---|---|---|---|
| `mlp_concat` k=32 | 142k | 2.059 | 2.003 | 2.012 | 1.981 (60k) | 1.617 |
| `rnn` | 31k | 1.970 | 1.885 | 1.829 | 1.827 (60k) | 1.601 |
| `lstm` H=128 | 93k | 1.923 | 1.801 | 1.713 | 1.713 (80k, still falling) | 1.466 |
| `attention` 6L x 128d | 1.2M | 1.913 | 1.773 | 1.687 | 1.680 (72k) | 1.331 |
| `attention` 6L x 384d | 10.7M | 1.883 | 1.743 | 1.700 | 1.669 (52k) | 1.261 |

Prediction 1 partially refuted: one rank inversion, caused by overfitting - by final val the 1.2M transformer passes the 10.7M one (1.687 vs 1.700); by min val the order survives (1.680 vs 1.669). Concat's early-convergence clause confirmed (best ~1.96-1.98, flat-to-worse after 10-20k).
Prediction 2 REFUTED: the 10.7M model bottoms at 1.669 at iter 52k and then *rises* - it never reaches [1.50, 1.62] and **KN (1.634) survives 16x budget unbeaten** by every single model.
Prediction 3 confirmed: overfitting onset ordered exactly by params/data ratio - 384d turns up at 52k (train 1.26 vs val 1.70), 128d marginally at 72k, the small recurrent models end on their minima.
Prediction 4 confirmed: the LSTM-rnn gap compounds, 0.047 -> 0.084 -> 0.116.
Prediction 5 confirmed: lstm at 80k (5.7 TFLOPs) reaches 1.713, beating the 41-TFLOP attention-384 5k snapshot (1.883) at 7x less compute.

Compute frontier at 80k: lstm 1.713 @ 5.7 TFLOPs; attention-128 1.687 @ 74 TFLOPs; attention-384 1.669 @ 657 TFLOPs - 9x the compute of attention-128 buys 0.011 nats.

**Conclusion / next steps.**
Convergence reshuffles the podium (the LSTM and both transformers all pass the rnn's 5k-era crown) but the deepest at-budget conclusion stands: counting is still champion among single models, because unregularized neural scale overfits 892k characters before reaching the counting floor.
The missing ingredients for beating KN are now enumerable rather than mysterious: regularization (dropout/weight decay - bigram.py's 1.48 uses dropout 0.2), longer context (block 256 vs our 32), and a LR schedule - tokens alone were not the answer.
The compute frontier belongs to the recurrent models at the low end; capacity only pays when data (and regularization) can feed it.
Next candidates: a regularization entry (dropout + weight decay on the 6L transformers), context 256, and refreshing the KN mixture with the lstm@80k component (its 1.713 vs the 1.923 used previously should push the 1.567 mixture lower).

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

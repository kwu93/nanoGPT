# Research Journal

This file is the index and the living scoreboard; full write-ups live elsewhere.
`journal/page1.md` and `journal/page2.md` are the frozen pre-restructure log (2026-07-12 to 2026-07-16).
From here on, each experiment is a directory under `experiments/runs/` with its spec, config, results, and report, developed on its own branch and merged via PR.
The operating model is documented in `PROCESS.md`.

## Scoreboard

Val NLL in nats/char on tiny Shakespeare (892k train / 223k val chars, vocab 65).

### Reference scale (`ref-v2`: E64/H256, 2,048 tokens/step, wd 0.1, checkpoint-at-val-min)

Neural entries quote mean best_val_loss over 3 seeds.

| model | best at | val NLL | source |
|---|---|---|---|
| `mlp_sum`, wd 0.1 | k=3 | 2.269 | page 2, weight-decay entry |
| `mlp_concat`, wd 0.1 | k=5 | 1.772 | page 2, weight-decay entry |
| `mlp_concat`, wd 0.1 + dropout 0.2 | k=8 | **1.769** | page 2, dropout entry |
| Kneser-Ney backoff | 5 chars | **1.634** | page 1 |

KN's margin over the best neural window model is 0.135 nats, down from 0.267 at the page-1 standard setting.
Matched-scoring anatomy (val positions with full 32-char context only): concat k=8 scores 1.739 against KN depth-8's 1.659, and the gap concentrates almost entirely off-support (page 2, KN-anatomy entry).

### Standard setting (`std-v1`: E32/H128, ~128 tokens/step, 5k iters) - frozen 2026-07-13

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

At 16x budget (80k iters, seed 0): `lstm` 1.713 (still falling), `attention` 6L x 128d 1.687, `attention` 6L x 384d 1.669 at its minimum (overfits after iter 52k).

## Experiment index

One row per experiment under the new structure; the id links to the experiment directory.

| id | date | question | verdict |
|---|---|---|---|
| - | - | (none yet) | - |

Legacy entries: 21 on page 1 and 4 on page 2; their section headers are the index.

## Key learnings

The distilled list (16 items, still current) lives at `journal/page1.md`, section "Key learnings so far".
Cost accounting (storage / compute / data budgets and how winners flip) is in the same file.

## Standing next steps

- Lambda sweep ({0.03, 0.3} around wd 0.1) at k=5/8 to find where regularization stops paying.
- The causal off-support test: train concat k=8 with random context truncation; if the d<=3 bucket gap closes materially, the KN-anatomy diagnosis is confirmed.
- Score the existing attention checkpoints per-d-bucket for the same off-support signature.
- Refresh the LSTM+KN mixture with per-bucket weights now that we know where each model wins.
- rnn / lstm / attention scale-up under the regularized `ref-v2` protocol.

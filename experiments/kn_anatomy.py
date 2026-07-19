"""Tables A-D for the 'Anatomy of the KN gap' entry (JOURNAL_2.md).

Decomposes the KN-vs-mlp_concat gap at k=8 on a shared prediction set (val
positions i >= 32): (A) coverage of val contexts by train at each depth,
(B) NLL conditioned on the longest train-matched context suffix, (C) NLL
conditioned on the train count of the full 8-char context, (D) the depth at
which KN's back-off actually grounds when offered 32 characters.
Compares the depth-8 KN model against the mean of the three weight-decay
mlp_concat k=8 checkpoints recorded in runs.jsonl.
"""
import json
from collections import Counter

import numpy as np
import torch

from .ngram import KneserNeyModel
from .train import BASE_CONFIG, load_data, load_model

K = 8
LADDER = (1, 2, 3, 4, 5, 8, 16, 32)
CONCAT = dict(model='mlp_concat', context_k=K, dim_embed=64, dim_hidden=256,
              seq_len=32, batch_size=64, num_iters=10000, weight_decay=0.1)


def main():
    data, chars = load_data(BASE_CONFIG)
    V = len(chars)
    train_ids = data['train'].tolist()
    val_ids = data['val'].tolist()
    train_b, val_b = bytes(train_ids), bytes(val_ids)
    scored = list(range(32, len(val_ids)))
    print(f'train {len(train_ids):,} chars, val {len(val_ids):,}, scored {len(scored):,}')

    # longest matched suffix depth d per scored position: the largest d <= 32
    # such that the d chars before the position occur anywhere in train
    d_arr = np.zeros(len(val_ids), dtype=np.int32)
    alive = scored
    for d in range(1, 33):
        seen = set(train_b[j:j + d] for j in range(len(train_b) - d + 1))
        alive = [i for i in alive if val_b[i - d:i] in seen]
        for i in alive:
            d_arr[i] = d
        if not alive:
            break

    # table A: coverage per ladder depth (+ count stats, reported in prose)
    counts_8 = None
    print(f'\nA. coverage of scored val positions by train contexts\n'
          f"{'k':>3} {'coverage':>9} {'mean cnt':>9} {'med':>5} {'singleton':>9}")
    for k in LADDER:
        cnt = Counter(train_b[j:j + k] for j in range(len(train_b) - k + 1))
        matched = [cnt[val_b[i - k:i]] for i in scored if d_arr[i] >= k]
        cov = len(matched) / len(scored)
        if matched:
            print(f'{k:>3} {cov:>9.1%} {np.mean(matched):>9.1f} '
                  f'{int(np.median(matched)):>5} {np.mean(np.array(matched) == 1):>9.1%}')
        else:
            print(f'{k:>3} {cov:>9.1%} {"-":>9} {"-":>5} {"-":>9}')
        if k == K:
            counts_8 = {i: cnt.get(val_b[i - K:i], 0) for i in scored}
        del cnt

    # per-position NLL: KN depth 8, built on train
    kn = KneserNeyModel(train_ids, K, V)
    kn_nll = np.array([-np.log(kn.prob(val_ids[i - K:i], val_ids[i], K)) for i in scored])
    print(f'\nKN k={K} aggregate on scored set: {kn_nll.mean():.4f}  (ladder value 1.659)')

    # per-position NLL: concat k=8, mean over the 3 weight-decay checkpoints
    paths = [r['model_path'] for line in open('runs.jsonl')
             for r in [json.loads(line)]
             if all(r['config'].get(f) == v for f, v in CONCAT.items())]
    assert len(paths) == 3, paths
    x = torch.tensor([val_ids[i - K:i] for i in scored], dtype=torch.long)
    y = torch.tensor([val_ids[i] for i in scored], dtype=torch.long)
    cc_nll = np.zeros(len(scored))
    with torch.no_grad():
        for p in paths:
            model, _ = load_model(p)
            for j in range(0, len(scored), 8192):
                logits, _ = model(x[j:j + 8192])
                lp = torch.log_softmax(logits[:, -1], dim=-1)
                cc_nll[j:j + 8192] += -lp.gather(-1, y[j:j + 8192, None])[:, 0].numpy()
    cc_nll /= len(paths)
    print(f'concat k={K} (wd, 3-seed mean) aggregate: {cc_nll.mean():.4f}')

    d_sc = d_arr[np.array(scored)]
    c_sc = np.array([counts_8[i] for i in scored])

    def bucket_table(title, labels, masks):
        print(f'\n{title}\n{"bucket":>8} {"share":>7} {"KN":>7} {"concat":>7} {"gap":>7}')
        for lab, m in zip(labels, masks):
            if m.sum():
                print(f'{lab:>8} {m.mean():>7.1%} {kn_nll[m].mean():>7.3f} '
                      f'{cc_nll[m].mean():>7.3f} {cc_nll[m].mean() - kn_nll[m].mean():>+7.3f}')

    dk = np.minimum(d_sc, K)
    bucket_table('B. NLL by longest matched suffix depth (gap = concat - KN)',
                 ['d=8', 'd=7', 'd=6', 'd=5', 'd=4', 'd<=3'],
                 [dk == 8, dk == 7, dk == 6, dk == 5, dk == 4, dk <= 3])

    bucket_table('C. NLL by train count of the full 8-char context',
                 ['c=0', 'c=1', 'c=2-4', 'c=5-19', 'c>=20'],
                 [c_sc == 0, c_sc == 1, (c_sc >= 2) & (c_sc <= 4),
                  (c_sc >= 5) & (c_sc <= 19), c_sc >= 20])

    print(f'\nD. effective grounding depth at nominal k=32: mean d = {d_sc.mean():.2f}')
    for j in (4, 5, 6, 8, 10, 12, 16, 20, 32):
        print(f'  P(d >= {j:>2}) = {(d_sc >= j).mean():>6.2%}')


if __name__ == '__main__':
    main()

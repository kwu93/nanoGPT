"""Count-based n-gram NLL floors.

An n-gram table predicts each character from the previous n-1 via smoothed
lookup: p(next | ctx) = (count(ctx, next) + alpha) / (count(ctx) + alpha * V).
Its val NLL is the floor for any model that sees only n-1 characters of
context, as long as the table is well estimated; once contexts get long enough
to be sparse in the training data, the table degrades and the val column stops
being a trustworthy floor (that failure is the motivation for neural LMs).
"""
import numpy as np

from .train import BASE_CONFIG, load_data


def build_table(ids, k):
    pair_counts, ctx_counts = {}, {}
    for i in range(k, len(ids)):
        ctx = tuple(ids[i - k:i])
        pair = (ctx, ids[i])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        ctx_counts[ctx] = ctx_counts.get(ctx, 0) + 1
    return pair_counts, ctx_counts


def table_nll(ids, table, k, vocab_size, alpha):
    pair_counts, ctx_counts = table
    nll = 0.0
    for i in range(k, len(ids)):
        ctx = tuple(ids[i - k:i])
        num = pair_counts.get((ctx, ids[i]), 0) + alpha
        den = ctx_counts.get(ctx, 0) + alpha * vocab_size
        nll -= np.log(num / den)
    return nll / (len(ids) - k)


if __name__ == '__main__':
    data, chars = load_data(BASE_CONFIG)
    train_ids, val_ids = data['train'].tolist(), data['val'].tolist()
    V = len(chars)
    print(f'{len(train_ids)} train chars, {len(val_ids)} val chars, vocab {V}')
    print(f"{'n':>2}  {'contexts':>9}  {'train nll a=0.01':>9}  {'val a=1':>8}  {'val a=0.01':>10}")
    for n in range(1, 9):
        k = n - 1
        table = build_table(train_ids, k)
        print(f'{n:>2}  {len(table[1]):>9}  '
              f'{table_nll(train_ids, table, k, V, alpha=0.01):>9.4f}  '
              f'{table_nll(val_ids, table, k, V, alpha=1.0):>8.4f}  '
              f'{table_nll(val_ids, table, k, V, alpha=0.01):>10.4f}')

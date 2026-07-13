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


def build_table(ids, k, key=tuple):
    """key canonicalizes the context window; the default keeps it ordered.
    Pass key=bag for an order-blind table (context = multiset of chars)."""
    pair_counts, ctx_counts = {}, {}
    for i in range(k, len(ids)):
        ctx = key(ids[i - k:i])
        pair = (ctx, ids[i])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        ctx_counts[ctx] = ctx_counts.get(ctx, 0) + 1
    return pair_counts, ctx_counts


def bag(window):
    return tuple(sorted(window))


def table_nll(ids, table, k, vocab_size, alpha, key=tuple):
    pair_counts, ctx_counts = table
    nll = 0.0
    for i in range(k, len(ids)):
        ctx = key(ids[i - k:i])
        num = pair_counts.get((ctx, ids[i]), 0) + alpha
        den = ctx_counts.get(ctx, 0) + alpha * vocab_size
        nll -= np.log(num / den)
    return nll / (len(ids) - k)


class BackoffModel:
    """Witten-Bell interpolated n-gram. Instead of one fixed context length,
    recursively blend the maximum-likelihood estimate at each length with the
    estimate one character shorter: p_k = lam * ML_k + (1 - lam) * p_{k-1},
    down to uniform. lam(ctx) = c(ctx) / (c(ctx) + d(ctx)), where d is the
    number of distinct continuations seen after ctx, so a context is trusted
    in proportion to its evidence and sparse levels fade out gracefully.
    Parameter-free: no alpha to tune."""

    def __init__(self, ids, max_context, vocab_size):
        self.vocab_size = vocab_size
        self.levels = []
        for k in range(max_context + 1):
            pair_counts, ctx_counts = build_table(ids, k)
            distinct = {}
            for ctx, _next in pair_counts:
                distinct[ctx] = distinct.get(ctx, 0) + 1
            self.levels.append((pair_counts, ctx_counts, distinct))

    def prob(self, window, next_id, depth=None):
        """window holds the most recent characters, oldest first."""
        p = 1.0 / self.vocab_size
        levels = self.levels if depth is None else self.levels[:depth + 1]
        for k, (pairs, ctxs, distinct) in enumerate(levels):
            ctx = tuple(window[len(window) - k:]) if k else ()
            c = ctxs.get(ctx, 0)
            if c == 0:
                continue
            lam = c / (c + distinct[ctx])
            p = lam * (pairs.get((ctx, next_id), 0) / c) + (1 - lam) * p
        return p


def backoff_nll(ids, model, depth):
    nll = 0.0
    for i in range(depth, len(ids)):
        nll -= np.log(model.prob(ids[i - depth:i], ids[i], depth))
    return nll / (len(ids) - depth)


class KneserNeyModel:
    """Interpolated Kneser-Ney. Two changes vs Witten-Bell: (1) absolute
    discounting - a flat D is subtracted from every count, so a singleton
    contributes at most 1-D instead of half its level's mass; (2) lower orders
    are built on continuation counts (in how many distinct longer contexts
    does this (ctx, next) appear) rather than raw frequency, so the back-off
    distribution answers "how plausible is this char in a novel context".
    D per level via the standard estimate n1 / (n1 + 2 n2)."""

    def __init__(self, ids, max_context, vocab_size):
        self.vocab_size = vocab_size
        raw = [build_table(ids, k) for k in range(max_context + 1)]
        # raw[k] serves as the top level when evaluating at depth k
        self.raw = [self._level(pairs, ctxs) for pairs, ctxs in raw]
        # cont[k] serves as a lower level; derived from raw level k+1
        self.cont = []
        for k in range(max_context):
            cont_pairs = {}
            for ctx, w in raw[k + 1][0]:
                key = (ctx[1:], w)
                cont_pairs[key] = cont_pairs.get(key, 0) + 1
            cont_ctxs = {}
            for (ctx, _w), c in cont_pairs.items():
                cont_ctxs[ctx] = cont_ctxs.get(ctx, 0) + c
            self.cont.append(self._level(cont_pairs, cont_ctxs))

    @staticmethod
    def _level(pairs, ctxs):
        distinct = {}
        for ctx, _w in pairs:
            distinct[ctx] = distinct.get(ctx, 0) + 1
        n1 = sum(1 for c in pairs.values() if c == 1)
        n2 = sum(1 for c in pairs.values() if c == 2)
        discount = n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.5
        return pairs, ctxs, distinct, discount

    def prob(self, window, next_id, depth):
        p = 1.0 / self.vocab_size
        for j in range(depth + 1):
            pairs, ctxs, distinct, D = self.raw[j] if j == depth else self.cont[j]
            ctx = tuple(window[len(window) - j:]) if j else ()
            total = ctxs.get(ctx, 0)
            if total == 0:
                continue
            c = pairs.get((ctx, next_id), 0)
            p = max(c - D, 0) / total + (D * distinct[ctx] / total) * p
        return p


if __name__ == '__main__':
    data, chars = load_data(BASE_CONFIG)
    train_ids, val_ids = data['train'].tolist(), data['val'].tolist()
    V = len(chars)
    print(f'{len(train_ids)} train chars, {len(val_ids)} val chars, vocab {V}')
    backoff = BackoffModel(train_ids, 7, V)
    kn = KneserNeyModel(train_ids, 7, V)
    print(f"{'n':>2}  {'contexts':>9}  {'train nll a=0.01':>9}  {'val a=1':>8}  {'val a=0.01':>10}  {'val backoff':>11}  {'val KN':>7}")
    for n in range(1, 9):
        k = n - 1
        table = build_table(train_ids, k)
        print(f'{n:>2}  {len(table[1]):>9}  '
              f'{table_nll(train_ids, table, k, V, alpha=0.01):>9.4f}  '
              f'{table_nll(val_ids, table, k, V, alpha=1.0):>8.4f}  '
              f'{table_nll(val_ids, table, k, V, alpha=0.01):>10.4f}  '
              f'{backoff_nll(val_ids, backoff, k):>11.4f}  '
              f'{backoff_nll(val_ids, kn, k):>7.4f}')

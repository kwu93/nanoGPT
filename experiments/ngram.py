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


DEEP_DEPTHS = (1, 2, 3, 4, 5, 8, 16, 32)


def deep_ladder(depths=DEEP_DEPTHS, alpha=0.01, config=BASE_CONFIG, verbose=True):
    """Train/val NLL for the fixed-alpha table, Witten-Bell, and Kneser-Ney
    at each context length in depths, including depths where materializing
    every backoff level at once would blow memory (k=32 needs 33 levels).

    Both backoff models fold lower orders into the estimate one level at a
    time, so instead of keeping all levels alive we stream them: build level
    k's counts, update a running per-position probability for every scored
    position, record NLLs if k is a requested depth, then free the level.
    WB applies each level directly; KN keeps its lower-order continuation
    chain q running and applies the raw top level only at requested depths
    (cont level k-1 is derived from raw level k, so one build serves both).
    Tables are bytes-keyed (vocab < 256). Every split is scored on positions
    i >= max(depths) so all cells share one prediction set; earlier journal
    entries scored i >= k, which shifts values by < 0.005.
    """
    from math import log

    data, chars = load_data(config)
    V = len(chars)
    max_k = max(depths)
    splits = {name: bytes(data[name].tolist()) for name in ('train', 'val')}
    train_b = splits['train']
    chains = {name: {'wb': [1.0 / V] * (len(b) - max_k), 'q': [1.0 / V] * (len(b) - max_k)}
              for name, b in splits.items()}
    results = {k: {} for k in depths}

    for k in range(max_k + 1):
        pairs = {}
        for i in range(k, len(train_b)):
            key = train_b[i - k:i + 1]
            pairs[key] = pairs.get(key, 0) + 1
        ctxs, distinct = {}, {}
        for key, c in pairs.items():
            ctx = key[:-1]
            ctxs[ctx] = ctxs.get(ctx, 0) + c
            distinct[ctx] = distinct.get(ctx, 0) + 1

        for name, b in splits.items():
            wb = chains[name]['wb']
            for idx in range(len(wb)):
                i = idx + max_k
                ctx = b[i - k:i]
                c = ctxs.get(ctx)
                if c is None:
                    continue
                lam = c / (c + distinct[ctx])
                wb[idx] = lam * (pairs.get(b[i - k:i + 1], 0) / c) + (1 - lam) * wb[idx]

        if k >= 1:
            cont_pairs = {}
            for key in pairs:
                ckey = key[1:]
                cont_pairs[ckey] = cont_pairs.get(ckey, 0) + 1
            cont_ctxs, cont_distinct = {}, {}
            for ckey, c in cont_pairs.items():
                ctx = ckey[:-1]
                cont_ctxs[ctx] = cont_ctxs.get(ctx, 0) + c
                cont_distinct[ctx] = cont_distinct.get(ctx, 0) + 1
            n1 = sum(1 for c in cont_pairs.values() if c == 1)
            n2 = sum(1 for c in cont_pairs.values() if c == 2)
            cont_D = n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.5
            for name, b in splits.items():
                q = chains[name]['q']
                for idx in range(len(q)):
                    i = idx + max_k
                    ctx = b[i - k + 1:i]
                    total = cont_ctxs.get(ctx)
                    if total is None:
                        continue
                    c = cont_pairs.get(b[i - k + 1:i + 1], 0)
                    q[idx] = max(c - cont_D, 0) / total + (cont_D * cont_distinct[ctx] / total) * q[idx]

        if k in depths:
            n1 = sum(1 for c in pairs.values() if c == 1)
            n2 = sum(1 for c in pairs.values() if c == 2)
            raw_D = n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.5
            for name, b in splits.items():
                wb, q = chains[name]['wb'], chains[name]['q']
                nll_a, nll_kn = 0.0, 0.0
                for idx in range(len(wb)):
                    i = idx + max_k
                    ctx = b[i - k:i]
                    c_pair = pairs.get(b[i - k:i + 1], 0)
                    total = ctxs.get(ctx)
                    nll_a -= log((c_pair + alpha) / ((total or 0) + alpha * V))
                    if total is None:
                        nll_kn -= log(q[idx])
                    else:
                        nll_kn -= log(max(c_pair - raw_D, 0) / total
                                      + (raw_D * distinct[ctx] / total) * q[idx])
                n = len(wb)
                results[k][name] = {
                    'alpha': nll_a / n,
                    'wb': -sum(map(log, wb)) / n,
                    'kn': nll_kn / n,
                }
            if verbose:
                r = results[k]
                print(f"k={k:>2}  contexts={len(ctxs):>9,}  "
                      f"alpha {r['train']['alpha']:.4f}/{r['val']['alpha']:.4f}  "
                      f"WB {r['train']['wb']:.4f}/{r['val']['wb']:.4f}  "
                      f"KN {r['train']['kn']:.4f}/{r['val']['kn']:.4f}  (train/val)",
                      flush=True)
        elif verbose:
            print(f"k={k:>2}  contexts={len(ctxs):>9,}  (level folded in)", flush=True)

    return results


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

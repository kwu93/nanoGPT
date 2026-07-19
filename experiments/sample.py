"""Sample text from every rung of the model ladder.

Val NLL scores one-step prediction with a gold history; generation feeds a
model its own output, so errors compound and one bad character can push the
history off the data manifold. This prints, from one shared primer and seed,
continuations from: uniform random, fixed n-gram tables (k = 0/1/2/4 chars of
context, alpha-smoothed), the alpha=0 ML 5-gram control, Witten-Bell and
Kneser-Ney backoff at the same depth, and the saved per-position mlp. Each
sample is labeled with the val NLL of the exact distribution being sampled.
"""
import numpy as np
import torch

from .ngram import BackoffModel, KneserNeyModel, backoff_nll, build_table, table_nll
from .train import BASE_CONFIG, decode, load_data, load_model

SEED = 0
N_CHARS = 400
PRIMER_LEN = 8
DEPTH = 4  # 5-gram = 4 chars of context
MLP_PATH = 'models/mlp-e32-h128-s0.pt'


def sample_table(table, k, vocab_size, alpha, primer, n, rng):
    pair_counts, ctx_counts = table
    ids = list(primer)
    for _ in range(n):
        ctx = tuple(ids[len(ids) - k:]) if k else ()
        den = ctx_counts.get(ctx, 0) + alpha * vocab_size
        if den == 0:  # alpha=0 in a novel context: no distribution left
            probs = np.full(vocab_size, 1.0 / vocab_size)
        else:
            probs = np.array([(pair_counts.get((ctx, j), 0) + alpha) / den
                              for j in range(vocab_size)])
        ids.append(int(rng.choice(vocab_size, p=probs)))
    return ids[len(primer):]


def sample_backoff(model, depth, vocab_size, primer, n, rng):
    ids = list(primer)
    for _ in range(n):
        window = ids[len(ids) - depth:]
        probs = np.array([model.prob(window, j, depth) for j in range(vocab_size)])
        assert abs(probs.sum() - 1) < 1e-6, probs.sum()
        ids.append(int(rng.choice(vocab_size, p=probs / probs.sum())))
    return ids[len(primer):]


def sample_neural(model, primer, n):
    torch.manual_seed(SEED)
    x = torch.tensor([primer], dtype=torch.long)
    return model.generate(x, n)[0].tolist()[len(primer):]


@torch.no_grad()
def neural_val_nll(model, val):
    """Exact mean NLL over sequential non-overlapping seq_len windows."""
    seq_len = model.config['seq_len']
    n = (len(val) - 1) // seq_len
    x = val[:n * seq_len].view(n, seq_len)
    y = val[1:n * seq_len + 1].view(n, seq_len)
    total = count = 0
    for i in range(0, n, 512):
        _, loss = model(x[i:i + 512], y[i:i + 512])
        total += loss.item() * x[i:i + 512].numel()
        count += x[i:i + 512].numel()
    return total / count


def main():
    data, chars = load_data(BASE_CONFIG)
    V = len(chars)
    train_ids, val_ids = data['train'].tolist(), data['val'].tolist()

    val_text = decode(val_ids, chars)
    primer = val_ids[val_text.index('\n\n'):][:PRIMER_LEN]
    print(f'primer (first turn boundary in val): {decode(primer, chars)!r}')

    tables = {k: build_table(train_ids, k) for k in (0, 1, 2, DEPTH)}
    wb = BackoffModel(train_ids, DEPTH, V)
    kn = KneserNeyModel(train_ids, DEPTH, V)
    mlp, mlp_chars = load_model(MLP_PATH)
    assert mlp_chars == chars

    def from_table(k, alpha):
        return lambda rng: sample_table(tables[k], k, V, alpha, primer, N_CHARS, rng)

    def from_backoff(model):
        return lambda rng: sample_backoff(model, DEPTH, V, primer, N_CHARS, rng)

    rows = [
        ('random (uniform)', np.log(V),
         lambda rng: rng.integers(0, V, N_CHARS).tolist()),
        ('unigram table (a=0.01)', table_nll(val_ids, tables[0], 0, V, 0.01),
         from_table(0, 0.01)),
        ('bigram table (a=0.01)', table_nll(val_ids, tables[1], 1, V, 0.01),
         from_table(1, 0.01)),
        ('trigram table (a=0.01)', table_nll(val_ids, tables[2], 2, V, 0.01),
         from_table(2, 0.01)),
        ('5-gram table (a=0.01)', table_nll(val_ids, tables[DEPTH], DEPTH, V, 0.01),
         from_table(DEPTH, 0.01)),
        ('5-gram table (a=0, ML control)', float('inf'),
         from_table(DEPTH, 0.0)),
        ('Witten-Bell backoff, depth 4', backoff_nll(val_ids, wb, DEPTH),
         from_backoff(wb)),
        ('Kneser-Ney backoff, depth 4', backoff_nll(val_ids, kn, DEPTH),
         from_backoff(kn)),
        (f'mlp (per-position, {MLP_PATH})', neural_val_nll(mlp, data['val']),
         lambda rng: sample_neural(mlp, primer, N_CHARS)),
    ]

    for name, nll, sampler in rows:
        text = decode(sampler(np.random.default_rng(SEED)), chars)
        print(f'\n=== {name}  |  val NLL {nll:.4f} ===')
        print(text)


if __name__ == '__main__':
    main()

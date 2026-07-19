import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import build_model

BASE_CONFIG = {
    'model': 'mlp',
    'data_path': 'input.txt',
    'test_ratio': 0.2,
    'seq_len': 8,
    'batch_size': 64,
    'dim_embed': 64,
    'dim_hidden': 256,
    'lr': 1e-3,
    'weight_decay': 0.1,
    'num_iters': 10000,
    'eval_every': 1000,
    'eval_iters': 100,
    'device': 'cpu',
}


def load_data(config):
    with open(config['data_path'], 'r') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split_idx = int(len(data) * (1 - config['test_ratio']))
    return {'train': data[:split_idx], 'val': data[split_idx:]}, chars


def encode(sentence, chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    return [stoi[x] for x in sentence]

def decode(indices, chars):
    return ''.join(chars[i] for i in indices)


def save_model(path, model, chars):
    """Persist a trained model as one self-contained file (config with
    vocab_size, vocab, weights); rebuild with load_model(path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'config': model.config,
        'chars': chars,
        'state_dict': model.state_dict(),
    }, path)


def load_model(path):
    payload = torch.load(path)
    model = build_model(payload['config'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, payload['chars']


@torch.no_grad()
def next_dist(model, context, chars):
    """Distribution over the next character after `context` (a non-empty
    string), as a {char: prob} dict. Feed a single character to compare
    against a bigram table's row."""
    context = context[-model.config['seq_len']:]
    stoi = {ch: i for i, ch in enumerate(chars)}
    x = torch.tensor([[stoi[ch] for ch in context]], dtype=torch.long)
    logits, _ = model(x)
    probs = F.softmax(logits[0, -1], dim=-1)
    return {ch: p.item() for ch, p in zip(chars, probs)}


def make_batch(data, config):
    seq_len, batch_size = config['seq_len'], config['batch_size']
    indices = np.random.randint(0, len(data) - seq_len, size=batch_size)
    X = torch.stack([data[i:i + seq_len] for i in indices])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in indices])
    device = config['device']
    return X.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(data, model, config):
    split_loss = {}
    model.eval()  # in case the model has dropout
    for split in ['train', 'val']:
        losses = []
        for _ in range(config['eval_iters']):
            X, y = make_batch(data[split], config)
            _, loss = model(X, y)
            losses.append(loss.item())
        split_loss[split] = float(np.mean(losses))
    model.train()
    return split_loss


def train(config, seed=0):
    """Run one training run; return final metrics, loss curves, and the model.
    The returned model carries the best-val weights seen at any eval point
    (checkpoint-at-val-min), located by best_val_loss/best_iter; train_loss
    and val_loss remain the final-iterate metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data, chars = load_data(config)
    config = {**config, 'vocab_size': len(chars)}
    model = build_model(config).to(config['device'])
    optim = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                              weight_decay=config['weight_decay'])

    metrics = defaultdict(list)
    best = {'val_loss': float('inf'), 'iter': 0, 'state_dict': None}

    def evaluate(i):
        loss = estimate_loss(data, model, config)
        metrics['iter'].append(i)
        metrics['train_loss'].append(loss['train'])
        metrics['val_loss'].append(loss['val'])
        if loss['val'] < best['val_loss']:
            best.update(val_loss=loss['val'], iter=i,
                        state_dict=copy.deepcopy(model.state_dict()))

    for i in range(config['num_iters']):
        if i % config['eval_every'] == 0:
            evaluate(i)
        X, y = make_batch(data['train'], config)
        _, batch_loss = model(X, y)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    evaluate(config['num_iters'])
    model.load_state_dict(best['state_dict'])

    return {
        'train_loss': metrics['train_loss'][-1],
        'val_loss': metrics['val_loss'][-1],
        'best_val_loss': best['val_loss'],
        'best_iter': best['iter'],
        'num_params': sum(p.numel() for p in model.parameters()),
        'curves': dict(metrics),
        'model': model,
        'chars': chars,
    }


if __name__ == '__main__':
    result = train(BASE_CONFIG)
    print(f"params={result['num_params']}  "
          f"train_loss={result['train_loss']:.4f}  val_loss={result['val_loss']:.4f}  "
          f"best_val_loss={result['best_val_loss']:.4f} @ iter {result['best_iter']}")
    context = torch.zeros((1, 1), dtype=torch.long, device=BASE_CONFIG['device'])
    sample = result['model'].generate(context, max_new_tokens=300)
    print(decode(sample[0].tolist(), result['chars']))

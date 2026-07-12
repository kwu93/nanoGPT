from collections import defaultdict

import numpy as np
import torch

from .model import build_model

BASE_CONFIG = {
    'model': 'mlp',
    'data_path': 'input.txt',
    'test_ratio': 0.2,
    'seq_len': 8,
    'batch_size': 4,
    'dim_embed': 32,
    'dim_hidden': 128,
    'lr': 1e-3,
    'num_iters': 5000,
    'eval_every': 250,
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


def decode(indices, chars):
    return ''.join(chars[i] for i in indices)


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
    """Run one training run; return final metrics, loss curves, and the model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data, chars = load_data(config)
    config = {**config, 'vocab_size': len(chars)}
    model = build_model(config).to(config['device'])
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'])

    metrics = defaultdict(list)

    def evaluate(i):
        loss = estimate_loss(data, model, config)
        metrics['iter'].append(i)
        metrics['train_loss'].append(loss['train'])
        metrics['val_loss'].append(loss['val'])

    for i in range(config['num_iters']):
        if i % config['eval_every'] == 0:
            evaluate(i)
        X, y = make_batch(data['train'], config)
        _, batch_loss = model(X, y)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    evaluate(config['num_iters'])

    return {
        'train_loss': metrics['train_loss'][-1],
        'val_loss': metrics['val_loss'][-1],
        'num_params': sum(p.numel() for p in model.parameters()),
        'curves': dict(metrics),
        'model': model,
        'chars': chars,
    }


if __name__ == '__main__':
    result = train(BASE_CONFIG)
    print(f"params={result['num_params']}  "
          f"train_loss={result['train_loss']:.4f}  val_loss={result['val_loss']:.4f}")
    context = torch.zeros((1, 1), dtype=torch.long, device=BASE_CONFIG['device'])
    sample = result['model'].generate(context, max_new_tokens=300)
    print(decode(sample[0].tolist(), result['chars']))

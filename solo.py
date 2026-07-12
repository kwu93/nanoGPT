import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
import torch
from collections import defaultdict

with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])

config = {
    'seq_len': 8,
    'batch_size': 4,
    'vocab_size': vocab_size,
    'dim_embed': 32,
    'eval_iters': 1000,
    'dim_hidden': 128,
    'num_iters': 5000,
    'print_every': 100,
    'test_ratio': 0.2,
    'lr': 0.001,


}


def train_test_split(data, test_ratio):
    split_idx = int(len(data) * (1 - test_ratio))
    train_data, val_data = data[:split_idx], data[split_idx:]
    return train_data, val_data

def create_data_dictionary(text):
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, test_data = train_test_split(data, test_ratio=0.2)
    return {
        'train': train_data,
        'val': test_data,
    }

def make_batch(data, config):
    seq_len, batch_size = config['seq_len'], config['batch_size']
    indices = np.random.randint(0, len(data) - seq_len, size=batch_size)
    X = torch.stack([data[i:i+seq_len] for i in indices])
    y = torch.stack([data[i+1:i+seq_len+1] for i in indices])
    return X, y


@torch.no_grad()
def estimate_loss(data, model, config):
    eval_iters = config['eval_iters']
    split_loss = {}
    model.eval() # In case the model has drop-out. 
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, y = make_batch(data[split], config)
            _, loss = model(X, y)
            losses.append(loss)
        split_loss[split] = np.mean(losses)
    model.train()
    return split_loss


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        dim_embed, vocab_size, seq_len, dim_hidden = config['dim_embed'], config['vocab_size'], config['seq_len'], config['dim_hidden']
        self.token_embedding = nn.Embedding(vocab_size, dim_embed)
        self.pos_embedding = nn.Embedding(seq_len, dim_embed)
        self.block = nn.Sequential(
            nn.Linear(dim_embed, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, vocab_size)
        )
    def forward(self, x, y=None):
        B, T = x.shape
        x = self.token_embedding(x)
        pos = self.pos_embedding(torch.arange(T))
        x = x + pos
        logits = self.block(x)
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        return logits, loss
    
    def generate(self, x, max_new_tokens):
        seq_len = config['seq_len']
        # Here X has shape (B, T , C)
        for _ in range(max_new_tokens):
            context = x[:, -seq_len:]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

data = create_data_dictionary(text)
model = Model(config)
optim = torch.optim.Adam(params = model.parameters(), lr=config['lr'])

metrics = defaultdict(list)
for i in range(config['num_iters']):
    if i % config['print_every'] == 0:
        loss = estimate_loss(data, model, config)
        metrics['iter'].append(i)
        metrics['train_loss'].append(loss['train'])
        metrics['val_loss'].append(loss['val'])
    X, y = make_batch(data['train'], config)
    logits, batch_loss = model(X, y)

    optim.zero_grad()
    batch_loss.backward()
    optim.step()

plt.plot(metrics['iter'], metrics['train_loss'], label='train')
plt.plot(metrics['iter'], metrics['val_loss'], label='val')
plt.grid(); plt.legend();

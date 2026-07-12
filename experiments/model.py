import torch
import torch.nn as nn
from torch.nn import functional as F

MODELS = {}

def register(name):
    def deco(cls):
        MODELS[name] = cls
        return cls
    return deco


def build_model(config):
    return MODELS[config['model']](config)


@register('mlp')
class MLPModel(nn.Module):
    """Token + position embeddings fed through a two-layer MLP. No attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_embed, dim_hidden = config['dim_embed'], config['dim_hidden']
        vocab_size, seq_len = config['vocab_size'], config['seq_len']
        self.token_embedding = nn.Embedding(vocab_size, dim_embed)
        self.pos_embedding = nn.Embedding(seq_len, dim_embed)
        self.block = nn.Sequential(
            nn.Linear(dim_embed, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, vocab_size),
        )

    def forward(self, x, y=None):
        B, T = x.shape
        tok = self.token_embedding(x)
        pos = self.pos_embedding(torch.arange(T, device=x.device))
        logits = self.block(tok + pos)
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens):
        seq_len = self.config['seq_len']
        for _ in range(max_new_tokens):
            context = x[:, -seq_len:]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x

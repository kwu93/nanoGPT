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


class AutoregressiveModel(nn.Module):
    """Shared sampling loop: feed the last seq_len tokens, sample the next."""

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


@register('mlp')
class MLPModel(AutoregressiveModel):
    """Token + position embeddings fed through a two-layer MLP. Per-position:
    the prediction at position t sees only token t."""

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


@register('mlp_concat')
class ConcatMLPModel(AutoregressiveModel):
    """Bengio (2003)-style neural n-gram: at each position, concatenate the
    embeddings of the last context_k tokens and predict the next from that.
    Causal by construction (left zero-padding covers positions before the
    window). No position embedding: each relative offset owns its own slice of
    the first Linear's weights, which encodes position structurally."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_embed, dim_hidden = config['dim_embed'], config['dim_hidden']
        vocab_size, context_k = config['vocab_size'], config['context_k']
        self.token_embedding = nn.Embedding(vocab_size, dim_embed)
        self.block = nn.Sequential(
            nn.Linear(context_k * dim_embed, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, vocab_size),
        )

    def forward(self, x, y=None):
        B, T = x.shape
        k = self.config['context_k']
        emb = self.token_embedding(x)
        # slot j (oldest to newest) holds the embedding of the token j steps back
        shifted = [F.pad(emb, (0, 0, j, 0))[:, :T] for j in range(k - 1, -1, -1)]
        logits = self.block(torch.cat(shifted, dim=-1))
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        return logits, loss

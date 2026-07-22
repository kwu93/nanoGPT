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


def mlp_block(dim_in, dim_hidden, dim_out, dropout=0.0):
    """Shared two-layer MLP with optional dropout on the hidden activations.
    The Dropout module is only inserted when active so checkpoints saved
    before the dropout option existed keep their state_dict layout."""
    layers = [nn.Linear(dim_in, dim_hidden), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)


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
        self.block = mlp_block(dim_embed, dim_hidden, vocab_size,
                               config.get('dropout', 0.0))

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


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, seq_len):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)).bool())

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        hs = C // self.n_heads
        q = q.view(B, T, self.n_heads, hs).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hs).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * hs ** -0.5
        att = att.masked_fill(~self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, dim_hidden, n_heads, seq_len):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_heads, seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


@register('attention')
class AttentionModel(AutoregressiveModel):
    """bigram.py in miniature: token + position embeddings, n_layers of
    [causal multi-head attention + FFN] with pre-layernorm residuals, linear
    head. The mixing weights are activations computed per input rather than
    parameters, so context reach is the full window in one layer, at O(T^2)
    attention cost and with near-constant parameters in seq_len."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_embed, dim_hidden = config['dim_embed'], config['dim_hidden']
        vocab_size, seq_len = config['vocab_size'], config['seq_len']
        n_heads = config.get('n_heads', 4)
        n_layers = config.get('n_layers', 1)
        self.token_embedding = nn.Embedding(vocab_size, dim_embed)
        self.pos_embedding = nn.Embedding(seq_len, dim_embed)
        self.blocks = nn.ModuleList(
            Block(dim_embed, dim_hidden, n_heads, seq_len) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(dim_embed)
        self.head = nn.Linear(dim_embed, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        tok = self.token_embedding(x)
        pos = self.pos_embedding(torch.arange(T, device=x.device))
        h = tok + pos
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_f(h))
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        return logits, loss


@register('rnn')
class RNNModel(AutoregressiveModel):
    """Vanilla recurrent LM (Elman/Mikolov style): one shared cell updates a
    fixed-size hidden state per character, h_t = tanh(Wx e_t + Wh h_{t-1}),
    read out for the next-char prediction at every position. Shares weights
    across offsets like mlp_sum, but composition is non-commutative so order
    survives; parameter count is independent of context length, and context
    is unbounded in principle (capped at seq_len here by the batch window).

    Optional `context_k` (windowed recurrence): position t's state is rebuilt
    from h=0 over the last min(t+1, k) tokens only, so every prediction is
    structurally blind past k characters, matching mlp_concat's reach at the
    same k. Absent or None keeps the unbounded behavior above, and
    context_k >= seq_len coincides with it; no parameters change either way,
    so pre-existing checkpoints load unmodified."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_embed, dim_hidden = config['dim_embed'], config['dim_hidden']
        vocab_size = config['vocab_size']
        self.token_embedding = nn.Embedding(vocab_size, dim_embed)
        self.wx = nn.Linear(dim_embed, dim_hidden)
        self.wh = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.head = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        emb = self.token_embedding(x)
        k = self.config.get('context_k')
        if k is None:
            h = emb.new_zeros(B, self.wh.in_features)
            states = []
            for t in range(T):
                h = torch.tanh(self.wx(emb[:, t]) + self.wh(h))
                states.append(h)
            hidden = torch.stack(states, dim=1)
        else:
            # One k-token window per position (slot 0 = oldest), recurrence
            # run over all B*T windows at once: k loop steps instead of T,
            # each a matmul over B*T rows. Slots that would reach before the
            # sequence start are no-ops (state stays at its h=0 init), so a
            # short prefix is treated exactly as the unbounded path treats it.
            shifted = [F.pad(emb, (0, 0, j, 0))[:, :T] for j in range(k - 1, -1, -1)]
            windows = torch.stack(shifted, dim=2).view(B * T, k, -1)
            valid = (torch.arange(T, device=x.device).unsqueeze(1)
                     >= torch.arange(k - 1, -1, -1, device=x.device))
            valid = valid.expand(B, T, k).reshape(B * T, k)
            h = emb.new_zeros(B * T, self.wh.in_features)
            for s in range(k):
                h_next = torch.tanh(self.wx(windows[:, s]) + self.wh(h))
                h = torch.where(valid[:, s].unsqueeze(1), h_next, h)
            hidden = h.view(B, T, -1)
        logits = self.head(hidden)
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        return logits, loss


@register('lstm')
class LSTMModel(AutoregressiveModel):
    """Gated recurrence. Where the vanilla rnn folds input into state through
    one fixed tanh, the LSTM computes input-dependent multiplicative gates
    that route information: forget (what to erase from the cell), input (what
    to write), output (what to expose). The cell state c is an additive
    highway across time, which is also what tames vanishing gradients."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_embed, dim_hidden = config['dim_embed'], config['dim_hidden']
        vocab_size = config['vocab_size']
        self.token_embedding = nn.Embedding(vocab_size, dim_embed)
        self.gates = nn.Linear(dim_embed + dim_hidden, 4 * dim_hidden)
        self.head = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        H = self.head.in_features
        emb = self.token_embedding(x)
        h = emb.new_zeros(B, H)
        c = emb.new_zeros(B, H)
        states = []
        for t in range(T):
            i, f, g, o = self.gates(torch.cat([emb[:, t], h], dim=-1)).chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            states.append(h)
        logits = self.head(torch.stack(states, dim=1))
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        return logits, loss


@register('mlp_sum')
class SumMLPModel(AutoregressiveModel):
    """Order-blind ablation of ConcatMLPModel: sum the window's embeddings
    instead of concatenating. Every offset shares the same downstream weights,
    so the model sees only the multiset (bag) of the last context_k characters.
    Order is destroyed, including which character is most recent, and the
    parameter count does not grow with context_k."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_embed, dim_hidden = config['dim_embed'], config['dim_hidden']
        vocab_size = config['vocab_size']
        self.token_embedding = nn.Embedding(vocab_size, dim_embed)
        self.block = mlp_block(dim_embed, dim_hidden, vocab_size,
                               config.get('dropout', 0.0))

    def forward(self, x, y=None):
        B, T = x.shape
        k = self.config['context_k']
        emb = self.token_embedding(x)
        shifted = [F.pad(emb, (0, 0, j, 0))[:, :T] for j in range(k)]
        logits = self.block(sum(shifted))
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
        self.block = mlp_block(context_k * dim_embed, dim_hidden, vocab_size,
                               config.get('dropout', 0.0))

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

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


#### -------- Hyperparameters --------
batch_size = 32
learning_rate =  1e-3
context_window = 8
max_iters = 5000
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
eval_interval = 300
num_heads = 4

n_embed = 32

train_frac = 0.9

torch.manual_seed(1337)

#### -------- Prepare Data --------

with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])


#### -------- Build the Dataset --------

data = torch.tensor(encode(text), dtype=torch.long)

train_index = int(train_frac * len(data))

train_data = data[:train_index]
val_data = data[train_index:]   

def make_batch(split_data):
    sample_indices = torch.randint(low=0, high = len(split_data) - context_window, size=(batch_size, ))
    X = torch.stack([split_data[i: i + context_window] for i in sample_indices])
    y = torch.stack([split_data[i+1: i + context_window + 1] for i in sample_indices])
    return X.to(device), y.to(device)   

@torch.no_grad()
def estimate_loss():
    output = {}
    m.eval()
    for split in ['train', 'val']:
        split_data = train_data if split == 'train' else val_data
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = make_batch(split_data)
            _, loss = m(X, y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    m.train()
    return output


testX, testY = make_batch(train_data)


#### ---- Define language model ----

class Block(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForward(n_embed) # n_embed

    def forward(self, x):
        x = self.sa_head(x)
        x = self.ffn(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

        self.net = nn.Sequential(
            nn.Linear(size, size, bias=True),
            nn.ReLU(),
#            nn.Linear(size, size, bias=True),
        )
    
    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (B, T, C)
        self.pos_embedding_table = nn.Embedding(context_window, n_embed) # (B, T, C)
        # Try 1
#        self.sa_head = MultiHeadAttention(num_heads, n_embed // num_heads) # // Divide by num heads for equal params?
#        self.ffn = FeedForward(n_embed)
        # Try 2
        self.block = Block(num_heads)
        self.lm_head = nn.Linear(n_embed, vocab_size) # (B ,T, vocab size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        pos_embeddings = self.pos_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + pos_embeddings
        x = self.block(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_window:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)








class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((context_window, context_window))))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * self.head_size **-0.5

        # This bit to sub index tril was important. 
        # I was trying to do tril on the whole thing, but current length 
        # of context might be less than `context_window`.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        out = wei @ v # (B, T, H)
        return out 











    
#### Training procedure

m = BigramLanguageModel()

# TODO: 'mps'?
device = 'cuda' if torch.cuda.is_available() else 'cpu'

m = m.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)


for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    xb, yb = make_batch(train_data)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))




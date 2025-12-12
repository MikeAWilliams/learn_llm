"""
Transformer Language Model Library

This module contains the core components for building and training a character-level
Transformer language model (GPT-like architecture).

Classes:
    Head: Single self-attention head
    MultiHeadAttention: Multiple parallel self-attention heads
    FeedForward: Position-wise feed-forward network
    Block: Transformer block combining attention and feed-forward layers
    BigramLanguageModel: Complete Transformer language model

Functions:
    create_vocabulary: Create character-to-index mappings from text
    encode: Convert string to list of integers
    decode: Convert list of integers to string
    get_batch: Generate training/validation batches
    estimate_loss: Evaluate model loss on train/val sets
    get_device: Automatically select best available device (CUDA/MPS/CPU)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


def create_vocabulary(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, vocab_size, stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text]


def decode(indices, itos):
    return "".join([itos[i] for i in indices])


def get_batch(split, train_data, val_data, batch_size, block_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model, train_data, val_data, eval_iters, batch_size, block_size, device
):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(
                split, train_data, val_data, batch_size, block_size, device
            )
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout):
        """
        Initialize a single attention head.

        Args:
            head_size (int): Dimension of key, query, and value projections
            n_embd (int): Embedding dimension
            block_size (int): Maximum context length
            dropout (float): Dropout probability
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        v = self.value(x)  # (B,T,C)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        out = wei @ v  # (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        """
        Initialize multi-head attention.

        Args:
            num_heads (int): Number of attention heads
            head_size (int): Dimension of each head
            n_embd (int): Embedding dimension
            block_size (int): Maximum context length
            dropout (float): Dropout probability
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by non-linearity"""

    def __init__(self, n_embd, dropout):
        """
        Initialize feed-forward network.

        Args:
            n_embd (int): Embedding dimension
            dropout (float): Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        """
        Initialize a Transformer block.

        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            block_size (int): Maximum context length
            dropout (float): Dropout probability
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x = x + is a residual connection. It aids in training deep models
        # apply layer normalization to the input before self attention and before feed forward
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """Transformer-based character-level language model"""

    def __init__(
        self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device
    ):
        """
        Initialize the language model.

        Args:
            vocab_size (int): Size of the vocabulary
            n_embd (int): Embedding dimension
            block_size (int): Maximum context length
            n_head (int): Number of attention heads
            n_layer (int): Number of Transformer blocks
            dropout (float): Dropout probability
            device (str): Device to place model on
        """
        super().__init__()
        self.block_size = block_size
        self.device = device

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # position embedding table is a second embedding by position. Eg it will associate tokens at a given index together
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass through the model.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T)
            targets (torch.Tensor, optional): Target token indices of shape (B, T)

        Returns:
            tuple: (logits, loss)
                - logits: Predicted logits of shape (B, T, vocab_size)
                - loss: Cross-entropy loss (None if targets not provided)
        """
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.

        Args:
            idx (torch.Tensor): Starting context of shape (B, T)
            max_new_tokens (int): Number of tokens to generate

        Returns:
            torch.Tensor: Generated sequence of shape (B, T + max_new_tokens)
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

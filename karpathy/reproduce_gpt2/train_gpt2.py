import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# in the gpt from scratch video this was the MultiHeadAttention class
# this class does the same thing all in one set of matrix operations, which is more performant
# unfortunately I (Mike) am not following the math to really understand this is true
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # maximum sequence length
    vocab_size: int = (
        50257  # 256 possible bytes + 50,000 bpe merges + 1 special end document token
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# reproduce the gpt2 model structure
# plan is to load the weights and verify that we got it right
# use the hugging face data
# then train the weights ourselvs to prove we can do it
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme used by gpt2 and also attention is all you need
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # view the logits in 2d. Set the second dim to the last size of logits,
            # vocam_size. It figures out the first which will be B*T
            # then flatten out the targes into 1d
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # classmethod is python for c++ static. This method applies to the class not an object
    # Create and return a GPT based on model_type
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
        # -----------------------------------------------------------------------------


import tiktoken


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# -----------------------------------------------------------------------------
num_return_sequences = 5
max_length = 30


def get_device():
    result = "cpu"
    if torch.cuda.is_available():
        result = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        result = "mps"
    return result


def set_seed_on_device(device: str, seed: int):
    device_funct = {
        "cuda": torch.cuda.manual_seed_all,
        "mps": torch.mps.manual_seed,
        "cpu": torch.manual_seed,
    }[device]
    device_funct(seed)


def sync_on_device(device: str):
    device_funct = {
        "cuda": torch.cuda.synchronize,
        "mps": torch.mps.synchronize,
        "cpu": lambda: None,
    }[device]
    device_funct()


device = get_device()
print(f"found device {device}")

torch.manual_seed(1337)
set_seed_on_device(device, 1337)

import time

# adjust batch size to fit in the gpu
# before memory optmization 16 seems to be around 43 gigs
# so fine on my macbook, but way to big for my 5070ti
# each training iteration is about 6-7 seconds after the first couple which are closer to 15
# around 2k tokens per second
# after moving to tensor float 32 I am still only seeing 2k tokens per second
#
# lets try B = 8 no tensor float uses 26gb and gets 4k tok/sec
# with tensor_float I get the same as without. It seems that apple doesn't suport this
#
# B=8 runs out of memory without memory optimization on 5070ti
# B=4 uses 10,288 MiB without optimization, around 18,000 tok/sec
# With both autocast and high precision we use just a little less 10,058 MiB but we get ~26,000 tok/sec
# B=6 we use 13,916 MiB
train_loader = DataLoaderLite(B=6, T=1024)

torch.set_float32_matmul_precision("high")

# inline the forward pass and loss calculations
model = GPT(GPTConfig())
model.to(device)
# doens't seem to help on apple gpu
# model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # apple gpu doesn't seem to support this either I get no change in performance
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    # logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    sync_on_device(device)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_second = (train_loader.B * train_loader.T) / (t1 - t0)
    print(
        f"step {i}, loss {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_second:0.2f}"
    )


# stop the rest from executing
import sys

sys.exit(0)


model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)

# prefix tokens, kind of like a prompt
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# generate
torch.manual_seed(42)
set_seed_on_device(device, 42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # get the last column of logits
        logits = logits[:, -1, :]
        # convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # top k drops all but the top (50) probs by setting them to 0 and re-normalizing
        # this makes improbably tokens impossible
        # this is something hugging face does so we will do it here
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # pick an index out of probs
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
# note that the above will not match what hugging face generator produces
# Karpathy wasn't able to get that either. He wrote some similar code to the above
# to sample from the hugging face model directly and that matched for him. I am going to take his word for it

# return to
# https://youtu.be/l8pRSuU81PU?si=I_1Ahw2Z-i6h3jvm&t=6538

# build up a character level tokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()
print(f"lenght of dataset in characters: {len(text)}")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("all chars", "".join(chars))
print(f"vocab size: {vocab_size}")

stoi = {}
for i, ch in enumerate(chars):
    stoi[ch] = i

itos = {}
for i, ch in enumerate(chars):
    itos[i] = ch

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(encode("this will work"))
print(decode(encode("this will work")))

# print(encode("this will not work because there is no number 1 in the data set"))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
print(decode(data[:1000].tolist()))

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# demonstrate how prediction will work
# as we index through the characters in the block, the context is all the characters before and including the current one
# and the output is the next character
# so for the first character, the context is that character and the output is the second character
# but for the second character, the context is the first and second characters, and the output is the third character, and so on
# we batch these together into training blocks
# the following is all the training iterations in the first block
# this also has the benifit of training the model to predict based on a small context of one character up to the full block size context
# furthermore, because of this, the transformer can't work on context larger than the block size
block_size = 8
source = train_data[: block_size + 1]
expected_output = train_data[1 : block_size + 1]
for t in range(block_size):
    context_input = source[: t + 1]
    target = expected_output[t]
    print(f"when input is {context_input.tolist()} the target: {target}")

torch.manual_seed(1337)
block_size = 8  # what is the maximum context length for predictions


def get_batch(data, batch_size):
    # a tensor of size batch_size of random starting indices for the sequences
    # don't allow larger than len(data) - block_size so that there is room for the rest of the sequence
    start_index_tensor = torch.randint(len(data) - block_size, (batch_size,))
    # this is python list comprehension. for every start index i generate a list of chars starting at i and ending at i + block_size
    # torch.stack then combines the list of tensors into a single tensor
    input_tensor = torch.stack([data[i : i + block_size] for i in start_index_tensor])
    exptected_tensor = torch.stack(
        [data[i + 1 : i + block_size + 1] for i in start_index_tensor]
    )
    return input_tensor, exptected_tensor


batch_size = 4  # how many independent sequences will we process in parallel
xb, yb = get_batch(train_data, batch_size)
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

torch.manual_seed(1337)


class BigramLanguageModeler(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # B is batch size, t is block size, C is vocab size
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        # change logits and targets to 2d
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, next_idx), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModeler(vocab_size)
logits, loss = m(xb, yb)
print("logits shape:", logits.shape)
print("loss:", loss.item())

# generate from the model
idx = torch.zeros((1, 1), dtype=torch.long)
print(f"initial input is {decode(idx[0].tolist())}")  # \n
# we didn't train the model yet so we expect nonsense
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# train it

# not using gradient decent
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

print("\ntraining...")
batch_size = 32
for steps in range(10000):
    xb, yb = get_batch(train_data, batch_size)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("loss after training")
print(loss.item())
print("generating from trained model")
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=400)[0].tolist()))

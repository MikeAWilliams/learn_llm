import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random


# should be the same model as in 02_mlp.py but wrapped as a nn.Module
class MLP(nn.Module):
    def __init__(self, vocabulary_size, block_size, embed_dim, hidden_dim):
        super().__init__()
        self.block_size = block_size

        # was C
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)

        # replaces W1, b1
        self.linear1 = nn.Linear(block_size * embed_dim, hidden_dim, bias=True)

        # replaces W2, b2
        self.linear2 = nn.Linear(hidden_dim, vocabulary_size, bias=True)

    def forward(self, x):
        # x shape: (batch_size, block_size)
        emb = self.embedding(x)  # (batch_size, block_size, embed_dim)
        emb = emb.view(
            emb.shape[0], -1
        )  # flatten to (batch_size, block_size * embed_dim)
        h = torch.tanh(self.linear1(emb))  # (batch_size, hidden_dim)
        logits = self.linear2(h)  # (batch_size, vocabulary_size)
        return logits


# not bothering with gpu here since it didn't work well in the original
words = open("names.txt", "r").read().splitlines()
print(f"{len(words)} words")

# build the vocabulary of characters
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

# define model hyperparameters
vocabulary_size = len(itos)
block_size = 3
embed_dim = 10
hidden_dim = 200

print("vocabulary size", vocabulary_size)


def build_dataset(words):
    X, Y = [], []
    for w in words:

        # print(w)
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

model = MLP(vocabulary_size, block_size, embed_dim, hidden_dim)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params} parameters")

# optmizer performs gradient descent on all parameters of the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
g = torch.Generator().manual_seed(2147483647)

print("training...")
batch_size = 32
num_iterations = 200000
update_modulo = num_iterations // 20
training_flip = num_iterations // 2
start_time = time.time()
for i in range(num_iterations):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    logits = model(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])

    if i % update_modulo == 0:
        print(f"% complete {i/num_iterations}, loss {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    if i == training_flip:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()

end_time = time.time()
print(f"training took {end_time - start_time} seconds")
# evaluate on dev set
logits = model(Xdev)
loss = F.cross_entropy(logits, Ydev)
print(f"dev loss {loss.item()}")

print("generating names...")
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        x = torch.tensor([context])
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))

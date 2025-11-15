import torch
import torch.nn.functional as F
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
# GPU is inefficient for such a small model, switch to CPU
device = "cpu"
print(f"Using device: {device}")

words = open("names.txt", "r").read().splitlines()
print(f"{len(words)} words")

# build the vocabulary of characters
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

vocabulary_size = len(itos)
print("vocabulary size", vocabulary_size)
block_size = 3  # context length: how many characters do we take to predict the next one


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

    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    print(X.shape, Y.shape)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
# add an embedding layer C in the paper to embed each character into a smaller dimention space
# in this case each letter will be embedded into a 10 dimention vector
embed_dim = 10
C = torch.randn((vocabulary_size, embed_dim), generator=g).to(device)
# first layer of the network
# notice block size * 2 because each letter is now represented by a 2 dimention vector after C
# 100 is the number of neurons in the hidden layer
hidden_dim = 200
W1 = torch.randn((block_size * embed_dim, hidden_dim), generator=g).to(device)
b1 = torch.randn(hidden_dim, generator=g).to(device)
# second layer of the network
# from the 101 neurons in the hidden layer to 27 output classes (characters)
W2 = torch.randn((hidden_dim, vocabulary_size), generator=g).to(device)
b2 = torch.randn(vocabulary_size, generator=g).to(device)
parameters = [C, W1, b1, W2, b2]
print(f"{sum(p.nelement() for p in parameters)} parameters")

for p in parameters:
    p.requires_grad = True

print("training...")
batch_size = 512
batch_size = 32
num_iterations = 200000 // (batch_size // 32)
update_modulo = num_iterations // 20
training_flip = num_iterations // 2
start_time = time.time()
for i in range(num_iterations):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    emb = C[Xtr[ix]]  # not X but a batch X[ix]
    # view reshapes the embeding tensor to allign with the first layer weight matrix W1. Equivelent to concatenating the 2D embeddings
    h = torch.tanh(emb.view(-1, block_size * embed_dim) @ W1 + b1)
    logits = h @ W2 + b2
    # counts = logits.exp()
    # probs = counts / counts.sum(1, keepdim=True)
    # loss = -probs[torch.arange(32), Y].log().mean()
    # cross entropy is equivelent to the above 3 lines
    loss = F.cross_entropy(logits, Ytr[ix])
    if i % update_modulo == 0:
        print(f"% complete {i/num_iterations}: loss {loss.item()}")
    # print(loss.item())
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < training_flip else 0.01
    for p in parameters:
        p.data += -lr * p.grad

end_time = time.time()
print(f"training took {end_time - start_time} seconds, using {device}")

# evaluate the loss on the dev set
emb = C[Xdev]
h = torch.tanh(emb.view(-1, block_size * embed_dim) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print("dev loss:", loss.item())

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size  # initialize with all ...
    while True:
        emb = C[torch.tensor([context])].to(device)  # (1,block_size,d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1).cpu()
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print("".join(itos[i] for i in out))

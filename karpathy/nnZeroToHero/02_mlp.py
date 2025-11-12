import torch
import torch.nn.functional as F

words = open("names.txt", "r").read().splitlines()
print(f"{len(words)} words")

# build the vocabulary of characters
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

# build the dataset
block_size = 3  # context length: how many characters do we take to predict the output y
X, Y = [], []
for w in words:
    # print(w)
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print("".join(itos[i] for i in context), "->", itos[ix])
        context = context[1:] + [ix]  # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

g = torch.Generator().manual_seed(2147483647)
# add an embedding layer C in the paper to embed each character into a smaller dimention space
# in this case each letter will be embedded into a 2 dimention vector
C = torch.randn((27, 2), generator=g)
# first layer of the network
# notice block size * 2 because each letter is now represented by a 2 dimention vector after C
# 100 is the number of neurons in the hidden layer
W1 = torch.randn((block_size * 2, 100), generator=g)
b1 = torch.randn(100, generator=g)
# second layer of the network
# from the 101 neurons in the hidden layer to 27 output classes (characters)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
print(f"{sum(p.nelement() for p in parameters)} parameters")

for p in parameters:
    p.requires_grad = True

for _ in range(1000):
    # minibatch construct
    ix = torch.randint(0, X.shape[0], (32,), generator=g)
    emb = C[X[ix]]  # not X but a batch X[ix]
    # view reshapes the embeding tensor to allign with the first layer weight matrix W1. Equivelent to concatenating the 2D embeddings
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    # counts = logits.exp()
    # probs = counts / counts.sum(1, keepdim=True)
    # loss = -probs[torch.arange(32), Y].log().mean()
    # cross entropy is equivelent to the above 3 lines
    loss = F.cross_entropy(logits, torch.tensor(Y[ix]))
    print(loss.item())
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -0.1 * p.grad

# continue at https://youtu.be/TCH_1BHY58I?si=eJ-tQeieGicjOeFk&t=2794

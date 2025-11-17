import torch
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2  # batch size, time steps, channels
x = torch.randn(B, T, C)
print("input:", x.shape)

# represent previous tokens and their embeddings as an average
# bow means "bag of words" and indicates we are ignoring order
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1, :]  # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)
# the abnove sucks because it does each operation in a python loop
# we can do it all in tensor operations instead
torch.manual_seed(42)

# may need to come back to the video at https://youtu.be/kCc8FmEb1nY?si=zyvlHGJ8wFe0yZXS&t=2843
# but lets try and comment our way to this

# starting with a = all 1s a@b does sum of columns of b
a = torch.ones(3, 3)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print("a=", a)
print("--")
print("b=", b)
print("--")
print("c=", c)

print("------------------")

# if we change a to be lower triangular we get cumulative sum of rows of b
a = torch.tril(torch.ones(3, 3))
c = a @ b
print("a=", a)
print("--")
print("b=", b)
print("--")
print("c=", c)

print("------------------")
# now we can use this to get cumulative average by making a sum to 1
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
c = a @ b
print("a=", a)
print("--")
print("b=", b)
print("--")
print("c=", c)

# now use the trick to get the bag of words representation
# wei is short for weights
wei = torch.tril(torch.ones(T, T))
wei = wei / torch.sum(wei, 1, keepdim=True)

# B is infered
xbow2 = wei @ x  # (B, T, T) @ (B, T, C) --> (B, T, C)

# allclose compares two tensors for near equality
print("xbow allclose xbow2:", torch.allclose(xbow, xbow2))
print("xbow[0, 0]:", xbow[0, 0])
print("xbow2[0, 0]:", xbow2[0, 0])
print("difference:", torch.abs(xbow - xbow2).max())

# use softmax to do the same
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print("xbow allclose xbow3:", torch.allclose(xbow, xbow3))
print("xbow2 allclose xbow3:", torch.allclose(xbow2, xbow3))

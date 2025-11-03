# following along with the video where he is working in a jupiter notebook
# plan to work this into reusable code after I understand it

import torch
import string
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()
print(len(words))
# print(words[:10])

# a bigram model predicts a character based on a single input caracter


# learning about our data
# build a dictionary of bigrams and their counts
def count_as_dict():
    bigram_counts = {}
    for w in words:
        # we want to encode the idea that some characters are more likely to start or end a word
        # invent new fake characters <S> and <E> for start and end of word
        chs = ["<S>"] + list(w) + ["<E>"]
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    print(len(bigram_counts))
    sorted_bigrams = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
    print(sorted_bigrams[:10])


# 27 is unique english letters + a special start and end character (which is the same for both)
counts = torch.zeros((27, 27), dtype=torch.int32)
chars = list(string.ascii_lowercase)
stoi = {ch: i for i, ch in enumerate(chars, start=1)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}
# print(stoi)
# print(itos)
# this time lets build up the counts
for w in words:
    # we want to encode the idea that some characters are more likely to start or end a word
    # invent new fake characters <S> and <E> for start and end of word
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        counts[ix1, ix2] += 1


def learn_about_probability_and_sampling():
    # counts[0] is the count of all bigrams that start with "."
    print(counts[0])

    # we want to convert the counts into probabilities
    # just work with the first row for the starting character of the word
    probs = counts[0].float()
    probs /= probs.sum()
    print(probs)
    print(probs.sum())

    g = torch.Generator().manual_seed(2147483647)
    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
    print(itos[ix])


# now lest generate some words
g = torch.Generator().manual_seed(2147483647)
print("\ngenerate 20 names\n")
for i in range(20):
    ix = 0
    out = []
    while True:
        p = counts[ix].float()
        p = p / p.sum()
        # find the next letter by sampling from the distribution
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        # if we found index 0 thats the . at the end of the word
        if ix == 0:
            break
        out.append(itos[ix])
    print("".join(out))


# stop here for now https://www.youtube.com/watch?v=PaCmpygFfXo?t=36m0s

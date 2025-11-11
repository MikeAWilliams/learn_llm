# following along with the video where he is working in a jupiter notebook
# plan to work this into reusable code after I understand it

import torch
import string
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


# convert a character to an index and back
# 27 is unique english letters + a special start and end character (which is the same for both)
chars = list(string.ascii_lowercase)
stoi = {ch: i for i, ch in enumerate(chars, start=1)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}


def develop_probability_based_model():
    # print(stoi)
    # print(itos)
    # this time lets build up the counts
    counts = torch.zeros((27, 27), dtype=torch.int32)
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
        ix = torch.multinomial(
            probs, num_samples=1, replacement=True, generator=g
        ).item()
        print(itos[ix])

    # normalize the counts to get probabilities
    probs = counts.float()
    # props.sum(dim=1) gives us a column vector of the sum of each row, keepdim makes it a column vector again so we can do the division instead of a simple array
    probs /= probs.sum(dim=1, keepdim=True)
    # prove that each row sums to 1
    print(probs.sum(dim=1))

    # now lest generate some words
    # the point of this is that even though we are sampling the probability distribution of characters we still don't get great names
    # Mostly because we are only looking at one character back
    g = torch.Generator().manual_seed(2147483647)
    print("\ngenerate 20 names\n")
    for i in range(5):
        ix = 0
        out = []
        while True:
            # p = counts[ix].float()
            # p = p / p.sum()
            p = probs[ix]
            # find the next letter by sampling from the distribution
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            # if we found index 0 thats the . at the end of the word
            if ix == 0:
                break
            out.append(itos[ix])
        print("".join(out))

    print("\nlooking at the quality of the model")
    bigrams_considered = 0
    log_likelyhood = 0.0
    # calculate the quality of the probability model
    # recall that the random likelyhood is 1/27 for any given character so prop > 0.037 is good
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            p = probs[ix1, ix2]
            # using log to transform converst 0-1 probabilities to -infinity to 0 space. 0 would be a perfect prediction
            # inverting the sign so that larger negative log likelihood is worse
            # but 0 is still perfect
            nll = -torch.log(p)
            # print(f"{ch1}{ch2}: {p:.4f} {nll:.4f}")
            bigrams_considered += 1
            log_likelyhood += nll
    # average negative log likelyhood is a loss function
    # result was 2.4541
    print(
        f"average negative log likelyhood per bigram: {log_likelyhood / bigrams_considered:.4f}"
    )

    # notice we could run the above on any given word
    # but words not in the training set, or more precicely words with letter pairs not in the training set will produce infinite loss, negative infinity
    # this sucks so people use model smoothing, which is adding a small count to every possible bigram so that no bigram has a 0 probability
    # this looks like prop = (count + 1).float() propb /= propb.sum(dim=1, keepdim=True)


# develop_probability_based_model()


def develop_nn_based_model():
    # x is input y output output is also called a label
    xs, ys = [], []
    for w in words:
        # we want to encode the idea that some characters are more likely to start or end a word
        # invent new fake characters <S> and <E> for start and end of word
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            # print(ch1, ch2)
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    # one hot encoding takes a list of integers and creates a list of vecotrs
    # each vector will be size 27 and 0s in all positions except for a 1 in the index corresponding to the character index
    g = torch.Generator().manual_seed(2147483647)
    # weights are 27x27 to produce 27 possible outputs for each of the 27 possible inputs
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    # gradient decent
    for k in range(500):
        # forward pass
        xenc = F.one_hot(xs, num_classes=27).float()
        # define the output of the input * weights as log of the counts (from above), called logits
        logits = xenc @ W
        # exponentiate the logits to get what we are defining as counts
        counts = logits.exp()
        # normalize the counts to get probabilities
        probs = counts / counts.sum(1, keepdim=True)
        # this is called a softmax operation and is used as a normaliztion
        # compute the loss, still on the first word for now
        loss = -probs[torch.arange(len(ys)), ys].log().mean()
        print(f"loss {loss.item()} at step {k}")

        # backward pass
        W.grad = None
        # this is pytorch magic. It tracked all the operations as it computed loss so it can now compute the gradient of loss with respect to W
        loss.backward()

        # update the weights
        W.data += -50 * W.grad
    # we expect the loss to be about 2.4 like the probabilistic model above
    # we could sample this model the same as the probabilistic model above, and it would produce the same results


develop_nn_based_model()

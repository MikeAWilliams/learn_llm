# following along with the video where he is working in a jupiter notebook
# plan to work this into reusable code after I understand it

words = open("names.txt", "r").read().splitlines()
print(len(words))
print(words[:10])

# a bigram model predicts a character based on a single input caracter
# build a dictionary of bigrams and their counts
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

with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()
print(f"lenght of dataset in characters: {len(text)}")
# print("first 1000 chars\n", text[:1000])
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

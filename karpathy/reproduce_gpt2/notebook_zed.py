# using zed REPL to be like jupyter notebook. # %% means start a new cell
# %% import from hugging face
from ast import increment_lineno

from PIL.ImageFont import MAX_STRING_LENGTH
from transformers import GPT2LMHeadModel

# %% explore the shape of the model
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")  # note this is the 124M version
sd_hf = model_hf.state_dict()  # dict k,v pairs are name, tensor for the model

for k, v in sd_hf.items():
    print(k, v.shape)
    # the first two rows of this are pretty interesting
    # transformer.wte.weight torch.Size([50257, 768])
    # transformer.wpe.weight torch.Size([1024, 768])
    # wte = weight token embedding it is 50257 X 768 because GPT2 has vocab size 50257 and 768 dimensional embedding
    # wpe = weight position embedding. 1024 X 768 because GPT has a context window of 1024 tokens
# %% make a quick plot of wpe
import matplotlib.pyplot as plt

# %% matplotlib inline

plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")
# %% look at a couple of columns
# for a couple of different embedding dims (150, 200, 250) plot how it changes with position
plt.plot(sd_hf["transformer.wpe.weight"][:, 150])
plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
plt.plot(sd_hf["transformer.wpe.weight"][:, 250])
# %% look at a couple of rows
# for the first word in the context, plot how it changes with dimension
plt.plot(sd_hf["transformer.wpe.weight"][0])
# %% generate some text
from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2")
set_seed(42)
generator(
    "Hello, I'm a language model,",
    max_new_tokens=30,
    num_return_sequences=5,
    truncation=True,
)

# %% good old shakespere data

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
data = text[:1000]
print(data[:100])
# %% use the gpt2 tokenizer on the data
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)
print(tokens[:24])
# %% make batches out of the tokens
import torch

# its a nice trick to load one more token than you need and then
# exclude the last one from the input and the first one from the output
# recall that view is a reference back to the raw data
buf = torch.tensor(tokens[: 24 + 1])
x = buf[:-1].view(4, 6)
y = buf[1:].view(4, 6)
print(x)
print(y)

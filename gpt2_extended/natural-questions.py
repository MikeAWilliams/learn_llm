"""
Natural Questions dataset (sentence-transformers version)
https://huggingface.co/datasets/sentence-transformers/natural-questions
Downloads and tokenizes the data and saves to disk.
Run simply as:
$ python natural-questions.py

Dataset info: 100k question-answer pairs
Will save tokens to "natural_questions" directory as train.npy and val.npy
Follows a format of user: <some question>\nassistant: <some answer>
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
local_dir = "tune_natural_questions"
val_split_ratio = 0.01  # Use 1% for validation (1000 examples)

# create the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # end of text token

def format(doc):
    """Format a question-answer pair as a conversational prompt"""
    return f"user: {doc['query']}\nassistant: {doc['answer']}"

def tokenize(text):
    """Tokenizes a text string and returns a numpy array of uint16 tokens"""
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens, dtype=np.uint16)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )
    return tokens_np

# Load the dataset (non-streaming to get all data at once)
print("Loading Natural Questions dataset (100k examples)...")
nq = load_dataset("sentence-transformers/natural-questions", split="train")
print(f"Dataset loaded: {len(nq)} examples")

# Calculate split point
val_size = int(len(nq) * val_split_ratio)
train_size = len(nq) - val_size
print(f"Split: {train_size} train, {val_size} validation")

# Process all examples
print("\nTokenizing all examples...")
all_tokens = []
for idx, example in enumerate(tqdm(nq, desc="Processing")):
    formatted_text = format(example)
    tokens = tokenize(formatted_text)
    all_tokens.append(tokens)

# Concatenate all tokens
print("\nConcatenating tokens...")
train_tokens = np.concatenate(all_tokens[:train_size])
val_tokens = np.concatenate(all_tokens[train_size:])

print(f"\nTrain tokens: {len(train_tokens):,}")
print(f"Validation tokens: {len(val_tokens):,}")
print(f"Total tokens: {len(train_tokens) + len(val_tokens):,}")

# Save to disk
train_filename = os.path.join(DATA_CACHE_DIR, "train.npy")
val_filename = os.path.join(DATA_CACHE_DIR, "val.npy")

print(f"\nSaving train data to {train_filename}...")
np.save(train_filename, train_tokens)

print(f"Saving validation data to {val_filename}...")
np.save(val_filename, val_tokens)

print("\nDone! Dataset processed and saved.")

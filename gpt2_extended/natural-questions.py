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

def tokenize_with_mask(doc):
    """Tokenizes a document and creates a loss mask.

    Returns:
        tokens: numpy array of tokens
        mask: numpy array where 1 = compute loss, 0 = ignore
    """
    # Format the full text
    user_part = f"user: {doc['query']}\nassistant: "
    assistant_part = doc['answer']

    # Tokenize each part separately to know where assistant response starts
    user_tokens = [eot] + enc.encode_ordinary(user_part)
    assistant_tokens = enc.encode_ordinary(assistant_part)

    # Combine tokens
    all_tokens = user_tokens + assistant_tokens
    tokens_np = np.array(all_tokens, dtype=np.uint16)

    # Create mask: 0 for user part (including eot and "assistant: "), 1 for assistant response
    mask_np = np.zeros(len(all_tokens), dtype=np.uint8)
    mask_np[len(user_tokens):] = 1  # Only compute loss on assistant's response

    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )

    return tokens_np, mask_np

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
all_masks = []
for idx, example in enumerate(tqdm(nq, desc="Processing")):
    tokens, mask = tokenize_with_mask(example)
    all_tokens.append(tokens)
    all_masks.append(mask)

# Concatenate all tokens and masks
print("\nConcatenating tokens and masks...")
train_tokens = np.concatenate(all_tokens[:train_size])
val_tokens = np.concatenate(all_tokens[train_size:])
train_masks = np.concatenate(all_masks[:train_size])
val_masks = np.concatenate(all_masks[train_size:])

print(f"\nTrain tokens: {len(train_tokens):,}")
print(f"Validation tokens: {len(val_tokens):,}")
print(f"Total tokens: {len(train_tokens) + len(val_tokens):,}")

# Save to disk
train_tokens_file = os.path.join(DATA_CACHE_DIR, "train.npy")
val_tokens_file = os.path.join(DATA_CACHE_DIR, "val.npy")
train_masks_file = os.path.join(DATA_CACHE_DIR, "train_mask.npy")
val_masks_file = os.path.join(DATA_CACHE_DIR, "val_mask.npy")

print(f"\nSaving train data to {train_tokens_file}...")
np.save(train_tokens_file, train_tokens)
np.save(train_masks_file, train_masks)

print(f"Saving validation data to {val_tokens_file}...")
np.save(val_tokens_file, val_tokens)
np.save(val_masks_file, val_masks)

# Calculate and print mask statistics
train_mask_ratio = train_masks.sum() / len(train_masks)
val_mask_ratio = val_masks.sum() / len(val_masks)
print(f"\nMask statistics:")
print(f"  Train: {train_masks.sum():,} / {len(train_masks):,} tokens will be trained on ({train_mask_ratio:.1%})")
print(f"  Val:   {val_masks.sum():,} / {len(val_masks):,} tokens will be trained on ({val_mask_ratio:.1%})")

print("\nDone! Dataset processed and saved.")
print(f"Files: train.npy, train_mask.npy, val.npy, val_mask.npy")

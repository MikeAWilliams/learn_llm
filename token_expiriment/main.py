import time

import torch

from library import (
    BigramLanguageModel,
    create_vocabulary,
    decode,
    encode,
    estimate_loss,
    get_batch,
    get_device,
)
from tokenizer import CharBPETokenizer

# Hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def scrub_data(text):
    """Remove punctuation and convert to lowercase"""
    text = text.lower()
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz .\n")
    return "".join(c for c in text if c in allowed_chars)


def scrub_minimal(text):
    """Lowercase, keep only letters, spaces, periods, and newlines"""
    text = text.lower()
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz .\n")
    return "".join(c for c in text if c in allowed_chars)


def prepare_data(text, train_split=0.9, custom_tokenizer=None):
    if custom_tokenizer:
        # Use custom tokenizer (e.g., BPE)
        encoded = custom_tokenizer.encode(text)
        data = torch.tensor(encoded, dtype=torch.long)

        vocab_info = {
            "chars": list(custom_tokenizer.vocab.values()),
            "vocab_size": custom_tokenizer.vocab_size,
            "stoi": {custom_tokenizer.vocab[i]: i for i in custom_tokenizer.vocab},
            "itos": custom_tokenizer.vocab,
            "tokenizer": custom_tokenizer,
        }
    else:
        # Use character-level encoding
        chars, vocab_size, stoi, itos = create_vocabulary(text)
        data = torch.tensor(encode(text, stoi), dtype=torch.long)

        vocab_info = {
            "chars": chars,
            "vocab_size": vocab_size,
            "stoi": stoi,
            "itos": itos,
        }

    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_info


def train_model(model, optimizer, train_data, val_data, device):
    start_time = time.time()

    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(
                model, train_data, val_data, EVAL_ITERS, BATCH_SIZE, BLOCK_SIZE, device
            )

            elapsed_time = time.time() - start_time
            avg_time_per_step = elapsed_time / (iter + 1)
            estimated_total_time = avg_time_per_step * MAX_ITERS
            remaining_time = estimated_total_time - elapsed_time

            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | "
                f"avg: {avg_time_per_step:.3f}s/step | est. total: {estimated_total_time / 60:.1f}min | "
                f"remaining: {remaining_time / 60:.1f}min"
            )

        xb, yb = get_batch(
            "train", train_data, val_data, BATCH_SIZE, BLOCK_SIZE, device
        )

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"Training done. Time taken: {total_time:.2f} seconds")

    return total_time


def generate_text(model, itos, device, max_new_tokens=500):
    """
    Generate text with optional custom decoder.

    Args:
        model: Trained model
        itos: Integer to string mapping
        device: Device
        max_new_tokens: Number of tokens to generate
    """

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens=max_new_tokens)[
        0
    ].tolist()
    return decode(generated_indices, itos)


def run_scenario(
    scenario_name,
    input_file,
    output_file,
    data_scrubber=None,
    custom_tokenizer=None,
):
    print(f"Running Scenario: {scenario_name}")

    torch.manual_seed(1337)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading data...")
    text = load_data(input_file)

    if data_scrubber:
        print("Scrubbing data...")
        text = data_scrubber(text)

    train_data, val_data, vocab_info = prepare_data(
        text, custom_tokenizer=custom_tokenizer
    )
    print(f"vocab_size: {vocab_info['vocab_size']}")

    print("Initializing model...")
    model = BigramLanguageModel(
        vocab_size=vocab_info["vocab_size"],
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
        device=device,
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{num_params:.2f}M parameters")

    print("Creating Optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Training...")
    train_model(model, optimizer, train_data, val_data, device)

    print("Generating from the model...")
    if custom_tokenizer:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_indices = model.generate(context, max_new_tokens=1000)[0].tolist()
        generated_text = custom_tokenizer.decode(generated_indices)
    else:
        generated_text = generate_text(model, vocab_info["itos"], device, 1000)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print("\n" + "=" * 80)
    print("Generated text:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    print(f"\nOutput saved to: {output_file}")


def scenario_base():
    """Base scenario: raw character-level encoding with no preprocessing"""
    print("\n" + "*" * 60, "   Base Scenario   ", "*" * 60)
    run_scenario(
        scenario_name="Base (Raw Character-Level)",
        input_file="input.txt",
        output_file="output_base.txt",
    )


def scenario_scrubbed():
    """Scrubbed scenario: lowercase, no newlines, only periods"""
    print("\n" + "*" * 60, "   Scrubbed Scenario   ", "*" * 60)
    run_scenario(
        scenario_name="Scrubbed (Lowercase, No Newlines, Period Only)",
        input_file="input.txt",
        output_file="output_scrubbed.txt",
        data_scrubber=scrub_minimal,
    )


def scenario_bpe(num_merges=100):
    """BPE tokenization scenario with configurable merge count"""
    print("\n" + "*" * 60, f"   BPE Scenario ({num_merges} merges)   ", "*" * 60)

    # Load data to train tokenizer
    text = load_data("input.txt")

    # Get base vocabulary size
    base_chars = sorted(set(text))
    base_vocab_size = len(base_chars)
    target_vocab_size = base_vocab_size + num_merges

    print(f"Base vocabulary size: {base_vocab_size}")
    print(f"Target vocabulary size: {target_vocab_size} ({num_merges} merges)")

    # Train BPE tokenizer
    print("Training BPE tokenizer...")
    bpe_tokenizer = CharBPETokenizer()
    bpe_tokenizer.train(text, target_vocab_size)

    # Run scenario with BPE tokenizer
    run_scenario(
        scenario_name=f"BPE Tokenization ({num_merges} new tokens)",
        input_file="input.txt",
        output_file=f"output_bpe_{num_merges}.txt",
        custom_tokenizer=bpe_tokenizer,
    )


def experiment_tokenizer():
    """Experiment with BPE tokenizer to show raw vs tokenized output"""
    print("\n" + "=" * 80)
    print("TOKENIZER EXPERIMENT")
    print("=" * 80)

    print("\nLoading data...")
    text = load_data("input.txt")

    print("\n" + "=" * 80)
    print("RAW CHARACTER-LEVEL ENCODING")
    print("=" * 80)
    chars, vocab_size, stoi, itos = create_vocabulary(text)
    raw_encoded = encode(text, stoi)

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Encoded length: {len(raw_encoded)} tokens")

    print("\n" + "=" * 80)
    print("BPE TOKENIZATION")
    print("=" * 80)

    tokenizer = CharBPETokenizer()
    new_tokens = 500
    target_vocab = vocab_size + new_tokens

    print(f"\nTraining BPE tokenizer (target vocab size: {target_vocab})...")
    print("-" * 80)
    tokenizer.train(text, target_vocab)

    print("\n" + "-" * 80)
    print("Encoding sample with BPE...")
    bpe_encoded = tokenizer.encode(text)

    print(f"\nBPE encoded length: {len(bpe_encoded)} tokens")
    print(f"Compression ratio: {len(raw_encoded) / len(bpe_encoded):.2f}x")

    bpe_decoded = tokenizer.decode(bpe_encoded)

    print("\n" + "-" * 80)
    if text == bpe_decoded:
        print("✓ Round-trip successful: original == decoded")
    else:
        print("✗ Round-trip failed: original != decoded")

    print("\n" + "=" * 80)
    print("COMPLETE VOCABULARY")
    print("=" * 80)
    vocab_info = tokenizer.get_vocab_info()
    print(f"\nBase characters: {len(vocab_info['base_chars'])}")
    print(f"Number of merges: {vocab_info['num_merges']}")
    print(f"Final vocabulary size: {vocab_info['vocab_size']}")

    all_tokens = [vocab_info["vocab"][i] for i in sorted(vocab_info["vocab"].keys())]
    print(f"\nAll tokens: {all_tokens}")

    print("\n" + "=" * 80)


def main():
    scenario_bpe(num_merges=2000)


if __name__ == "__main__":
    main()

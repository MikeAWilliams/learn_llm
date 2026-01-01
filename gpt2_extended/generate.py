#!/usr/bin/env python
"""
Generate text from a GPT-2 checkpoint.

Usage:
    python generate.py --checkpoint log/model_05000.pt --prompt "Hello, I'm a language model,"
    python generate.py --checkpoint log/model_05000.pt --prompt "Once upon a time" --num_samples 8
"""

import argparse

import tiktoken
import torch
from torch.nn import functional as F

from model import GPT
from util import (
    get_device,
    load_checkpoint,
    set_seed_on_device,
    unwrap_model_state_dict,
)


def generate_text(model, prompt, enc, device, num_samples=4, max_length=32, seed=42):
    """
    Generate text continuations from a prompt.

    Args:
        model: The GPT model
        prompt: Text prompt to continue from
        enc: Tokenizer
        device: Device to run on
        num_samples: Number of samples to generate
        max_length: Maximum length of generated sequence
        seed: Random seed for reproducibility

    Returns:
        List of generated text strings
    """
    model.eval()

    # Encode the prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1)
    xgen = tokens.to(device)

    # Set up random number generator with same seed as training
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)

    # Generate tokens
    with torch.no_grad():
        while xgen.size(1) < max_length:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    # Decode the generated sequences
    generated_texts = []
    for i in range(num_samples):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        generated_texts.append(decoded)

    return generated_texts


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a GPT-2 checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., log/model_05000.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, I'm a language model",
        help="Text prompt to continue from",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate (default: 4)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="Maximum length of generated sequence (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42, same as training)",
    )
    args = parser.parse_args()

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    torch.manual_seed(1337)
    set_seed_on_device(device, 1337)

    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = checkpoint["config"]

    print(f"Creating model with vocab_size={model_config.vocab_size}")
    model = GPT(model_config)
    model.to(device)

    # Unwrap state dict if it has _orig_mod. prefix from torch.compile()
    state_dict = unwrap_model_state_dict(checkpoint["model"])
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.num_samples} samples...\n")
    print("=" * 70)

    generated_texts = generate_text(
        model,
        args.prompt,
        enc,
        device,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    # Display results
    for i, text in enumerate(generated_texts, 1):
        print(f"\nSample {i}:")
        print(text)
        print("-" * 70)

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()

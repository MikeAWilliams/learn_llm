"""
Interactive conversation script for GPT-2 fine-tuned models from Hugging Face.
Loads a model checkpoint from Hugging Face Hub and allows multi-turn conversations.

Usage:
    python conversation_hf.py --repo mikeawilliams/gpt2
    python conversation_hf.py --repo mikeawilliams/gpt2 --checkpoint model_19072-nq_00150.pt
"""

import argparse
import importlib.util
import sys

import tiktoken
import torch
from huggingface_hub import hf_hub_download


def load_model_from_hf(repo_id, checkpoint_name, device):
    """Load model from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repo ID (e.g., "mikeawilliams/gpt2")
        checkpoint_name: Name of checkpoint file (e.g., "model_19072-nq_00149.pt")
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"Downloading model code from {repo_id}...")
    model_code_path = hf_hub_download(repo_id=repo_id, filename="model.py")

    # Import the model module
    spec = importlib.util.spec_from_file_location("model", model_code_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    print(f"Downloading checkpoint {checkpoint_name}...")
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_name)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model
    model = model_module.GPT(checkpoint['config'])
    model.to(device)

    # Handle state dict with different prefixes
    state_dict = checkpoint['model']

    # Remove _orig_mod. prefix if present (from torch.compile)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

    # Remove module. prefix if present (from DDP)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model


def generate_response(model, prompt_tokens, max_new_tokens, device, enc, temperature=0.8, top_k=50):
    """Generate a response from the model given prompt tokens.

    Args:
        model: The GPT model
        prompt_tokens: Input tokens (list or tensor)
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on
        enc: Tokenizer
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter

    Returns:
        Generated text as string
    """
    model.eval()

    # Convert to tensor if needed
    if isinstance(prompt_tokens, list):
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    # Ensure it's 2D (batch dimension)
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Check if we've exceeded block size - if so, truncate from left
            if prompt_tokens.size(1) > model.config.block_size:
                prompt_tokens = prompt_tokens[:, -model.config.block_size:]

            # Forward pass
            logits, _ = model(prompt_tokens)

            # Get logits for last position
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample from the distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            prompt_tokens = torch.cat([prompt_tokens, next_token], dim=1)

            # Check if we hit end of text token
            if next_token.item() == enc._special_tokens.get("<|endoftext|>", -1):
                break

    return prompt_tokens.squeeze(0).tolist()


def extract_assistant_response(full_text, prompt_text):
    """Extract just the assistant's response from the full generated text.

    Args:
        full_text: Full generated text including prompt
        prompt_text: The original prompt

    Returns:
        Just the assistant's response
    """
    # Remove the prompt from the beginning
    if full_text.startswith(prompt_text):
        response = full_text[len(prompt_text):]
    else:
        response = full_text

    # Find the end of the assistant's response (next "user:" or end of text)
    if "\nuser:" in response:
        response = response.split("\nuser:")[0]

    # Remove trailing whitespace and newlines
    response = response.strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="Interactive conversation with fine-tuned GPT-2 from Hugging Face")
    parser.add_argument(
        "--repo",
        type=str,
        default="mikeawilliams/gpt2",
        help="Hugging Face repo ID (default: mikeawilliams/gpt2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_19072-nq_00149.pt",
        help="Checkpoint filename (default: model_19072-nq_00149.pt)",
    )
    parser.add_argument(
        "--useformat",
        action="store_true",
        help="If set, context will follow format user:<question>\\nassistant:<response>",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per response (default: 200)",
    )
    args = parser.parse_args()

    if args.useformat:
        print(f"Using assistant format")
    else:
        print(f"Using plain text format")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model from Hugging Face
    model = load_model_from_hf(args.repo, args.checkpoint, device)

    print(f"Model loaded successfully!")
    print(f"Block size: {model.config.block_size}")
    print()

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    # Conversation history (as tokens)
    conversation_tokens = []

    print("=" * 80)
    print("Interactive Conversation (type 'quit' or 'exit' to end)")
    print("=" * 80)
    print()

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Format as conversation turn
        user_turn = ""
        if args.useformat:
            user_turn = f"user: {user_input}\nassistant: "
        else:
            user_turn = f"{user_input}\n"

        # Tokenize the new user turn
        user_tokens = enc.encode_ordinary(user_turn)

        # Add to conversation history
        conversation_tokens.extend(user_tokens)

        # If conversation is too long, drop tokens from the beginning
        # Keep some buffer for generation
        max_context = model.config.block_size - args.max_tokens - 10
        if len(conversation_tokens) > max_context:
            # Drop from the beginning, but try to keep conversation boundaries
            tokens_to_drop = len(conversation_tokens) - max_context
            conversation_tokens = conversation_tokens[tokens_to_drop:]

        # Generate response
        print("Assistant: ", end="", flush=True)

        # Convert to tensor
        prompt_tensor = torch.tensor(conversation_tokens, dtype=torch.long, device=device)

        # Generate
        generated_tokens = generate_response(
            model,
            prompt_tensor,
            max_new_tokens=args.max_tokens,
            device=device,
            enc=enc,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Decode full text
        full_text = enc.decode(generated_tokens)

        # Extract just the assistant's response
        prompt_text = enc.decode(conversation_tokens)
        assistant_response = extract_assistant_response(full_text, prompt_text)

        print(assistant_response)
        print()

        # Add assistant's response to conversation history
        assistant_tokens = enc.encode_ordinary(assistant_response)
        conversation_tokens.extend(assistant_tokens)

        # Add a newline token to separate turns
        conversation_tokens.append(enc.encode_ordinary("\n")[0])


if __name__ == "__main__":
    main()

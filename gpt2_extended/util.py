"""Shared utility functions for GPT-2 training and inference."""

import os

import torch


def get_device():
    """Detect the best available device."""
    result = "cpu"
    if torch.cuda.is_available():
        result = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        result = "mps"
    return result


def set_seed_on_device(device: str, seed: int):
    """Set random seed for the specified device."""
    # Extract device type from strings like "cuda:0" -> "cuda"
    device_type = device.split(":")[0]

    device_funct = {
        "cuda": torch.cuda.manual_seed_all,
        "mps": torch.mps.manual_seed,
        "cpu": torch.manual_seed,
    }[device_type]
    device_funct(seed)


def load_checkpoint(checkpoint_path, device):
    """
    Load checkpoint from file.
    Returns: checkpoint_dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def unwrap_model_state_dict(state_dict):
    """
    Remove '_orig_mod.' prefix from state_dict keys.
    This prefix is added by torch.compile() but not needed for inference.

    Args:
        state_dict: Model state dictionary, possibly with _orig_mod. prefix

    Returns:
        Cleaned state dictionary without _orig_mod. prefix
    """
    unwrapped = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            # Remove the _orig_mod. prefix
            new_key = key[len("_orig_mod.") :]
            unwrapped[new_key] = value
        else:
            unwrapped[key] = value
    return unwrapped

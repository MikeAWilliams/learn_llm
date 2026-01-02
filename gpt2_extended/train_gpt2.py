import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from hellaswag import iterate_examples, render_example
from model import GPT, GPTConfig
from util import get_device, set_seed_on_device
from util import load_checkpoint as load_checkpoint_base

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Model configuration
    vocab_size: int = 50304  # Optimized for GPU efficiency (nearest multiple of 64)

    # Batch configuration
    # recal requirement total_batch_size % (B * T * ddp_world_size) == 0
    micro_batch_size: int = 8  # B - micro batch size per GPU
    sequence_length: int = 1024  # T - sequence length
    total_batch_size: int = 524288  # 2**19, ~0.5M tokens

    # Learning rate schedule
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 715
    max_steps: int = 19073

    # Optimizer parameters
    weight_decay: float = 0.1

    # Logging and checkpointing
    output_interval: int = 100
    checkpoint_interval: int = 5000
    log_dir: str = "log"

    # Data
    data_root: str = "edu_fineweb10B"

    # Random seed
    seed: int = 1337


# -----------------------------------------------------------------------------
# Data Loading Utilities
# -----------------------------------------------------------------------------


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """Lightweight data loader for training and validation."""

    def __init__(
        self, B, T, process_rank, num_processes, split, data_root, master_process
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {"train", "val"}
        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            print(f"found {len(self.shards)} shards for split {split}")
        self.reset()

    def reset(self):
        """Reset to the beginning of the dataset."""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """Get the next batch of data."""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def sync_on_device(device: str):
    """Synchronize the specified device."""
    # Extract device type from strings like "cuda:0" -> "cuda"
    device_type = device.split(":")[0]

    device_funct = {
        "cuda": torch.cuda.synchronize,
        "mps": torch.mps.synchronize,
        "cpu": lambda: None,
    }[device_type]
    device_funct()


def get_most_likely_row(tokens, mask, logits):
    """
    Helper function for HellaSwag eval.
    Takes tokens, mask, and logits, returns the index of the completion with the lowest loss.
    """
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # average loss for completion region only (where mask == 1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def get_lr(step, config):
    """Calculate learning rate with warmup and cosine decay."""
    min_lr = config.max_lr * config.min_lr_ratio
    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    if step > config.max_steps:
        return min_lr
    decay_ratio = (step - config.warmup_steps) / (
        config.max_steps - config.warmup_steps
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (config.max_lr - min_lr)


def format_time(seconds):
    """Format seconds into a human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------


def load_checkpoint(checkpoint_path, device):
    """
    Load checkpoint from file for training.
    Always resumes from step 0 with warmup learning rate.
    Returns: checkpoint_dict
    """
    checkpoint = load_checkpoint_base(checkpoint_path, device)
    print("Resuming training from step 0 with warmup learning rate")
    return checkpoint


def setup_ddp():
    """
    Set up Distributed Data Parallel (DDP).
    Returns: (ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process)
    """
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = get_device()
        print(f"using device: {device}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def evaluate_validation(model, val_loader, device, ddp, master_process, log_file, step):
    """Evaluate validation loss."""
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
    return val_loss_accum.item()


def evaluate_hellaswag(
    model, device, ddp, ddp_rank, ddp_world_size, master_process, log_file, step
):
    """Evaluate HellaSwag accuracy."""
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(
            num_correct_norm, dtype=torch.long, device=device
        )
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")
    return acc_norm


def generate_samples(model, device, ddp_rank, enc, master_process):
    """Generate text samples from the model."""
    model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)
    if master_process:
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


def save_checkpoint(raw_model, step, val_loss, log_dir, master_process):
    """Save model checkpoint."""
    if master_process:
        print("saving weights")
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            "model": raw_model.state_dict(),
            "config": raw_model.config,
            "step": step,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, checkpoint_path)


def train_step(model, train_loader, optimizer, device, ddp, grad_accum_steps):
    """Perform one training step with gradient accumulation."""
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_accum.item(), norm


def main(resume_from=None):
    config = TrainingConfig()
    # overrides for local testing
    # config.output_interval = 10
    # config.checkpoint_interval = 10
    # overrides for 80 gpu runs
    # set total batch size as a nice multiple of B * T. Picked 8 so I can run on 1,2,4,8 gpu
    # config.micro_batch_size = 96
    # config.total_batch_size = config.micro_batch_size * config.sequence_length * 8

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_ddp()

    torch.manual_seed(config.seed)
    set_seed_on_device(device, config.seed)

    enc = tiktoken.get_encoding("gpt2")

    B = config.micro_batch_size
    T = config.sequence_length
    assert config.total_batch_size % (B * T * ddp_world_size) == 0, (
        "make sure total_batch_size is divisible by B * T * ddp_world_size"
    )
    grad_accum_steps = config.total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {config.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train",
        data_root=config.data_root,
        master_process=master_process,
    )
    val_loader = DataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        data_root=config.data_root,
        master_process=master_process,
    )

    torch.set_float32_matmul_precision("high")

    # Load checkpoint if resuming (but always start from step 0)
    checkpoint_data = None
    if resume_from:
        checkpoint_data = load_checkpoint(resume_from, device)
        model_config = checkpoint_data["config"]
        if master_process:
            print(
                f"Loaded model config from checkpoint: vocab_size={model_config.vocab_size}"
            )
    else:
        model_config = GPTConfig(vocab_size=config.vocab_size)

    model = GPT(model_config)
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    raw_model = model.module if ddp else model

    # Load model weights if resuming (after compile to match checkpoint format)
    if checkpoint_data:
        state_dict = checkpoint_data["model"]
        # Handle DDP wrapper: checkpoint may have different prefix than current model
        # Checkpoint from non-DDP: has keys like "_orig_mod.transformer.wte.weight"
        # DDP model expects: "module._orig_mod.transformer.wte.weight"
        if ddp and not any(key.startswith("module.") for key in state_dict.keys()):
            # Add module. prefix for DDP
            state_dict = {f"module.{key}": value for key, value in state_dict.items()}
        elif not ddp and any(key.startswith("module.") for key in state_dict.keys()):
            # Remove module. prefix if loading DDP checkpoint into non-DDP model
            state_dict = {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
            }

        model.load_state_dict(state_dict)
        if master_process:
            print("Loaded model weights from checkpoint")

    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        device=device,
        master_process=master_process,
    )

    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(config.log_dir, "log.txt")

    with open(log_file, "w") as f:
        pass

    steps_completed = 0
    total_time_elapsed = 0.0
    avg_step_time = None

    if master_process and resume_from:
        print(f"Starting fresh training from step 0 (using weights from checkpoint)")

    for step in range(0, config.max_steps):
        t0 = time.time()
        last_step = step == config.max_steps - 1

        if step % config.output_interval == 0 or last_step:
            val_loss = evaluate_validation(
                model, val_loader, device, ddp, master_process, log_file, step
            )

            if step > 0 and (step % config.checkpoint_interval == 0 or last_step):
                save_checkpoint(
                    raw_model, step, val_loss, config.log_dir, master_process
                )

        if step % config.output_interval == 0 or last_step:
            evaluate_hellaswag(
                model,
                device,
                ddp,
                ddp_rank,
                ddp_world_size,
                master_process,
                log_file,
                step,
            )

        if (step > 0 and step % config.output_interval == 0) or last_step:
            generate_samples(model, device, ddp_rank, enc, master_process)

        loss, norm = train_step(
            model, train_loader, optimizer, device, ddp, grad_accum_steps
        )

        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        sync_on_device(device)
        t1 = time.time()
        dt = t1 - t0

        total_time_elapsed += dt
        steps_completed = step + 1
        tokens_processed = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        if step >= 10:
            if avg_step_time is None:
                avg_step_time = dt
            else:
                avg_step_time = 0.9 * avg_step_time + 0.1 * dt

        if master_process:
            steps_remaining = config.max_steps - steps_completed

            if avg_step_time is not None:
                estimated_time_remaining = steps_remaining * avg_step_time
                estimated_total_time = total_time_elapsed + estimated_time_remaining

                elapsed_str = format_time(total_time_elapsed)
                remaining_str = format_time(estimated_time_remaining)
                total_str = format_time(estimated_total_time)

                progress_pct = (steps_completed / config.max_steps) * 100

                print(
                    f"step {step:5d}/{config.max_steps} ({progress_pct:.1f}%) | "
                    f"loss: {loss:.6f} | lr {lr:.4e} | norm: {norm:.4f} | "
                    f"dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.0f} | "
                    f"elapsed: {elapsed_str} | remaining: {remaining_str} | total: {total_str}"
                )
            else:
                print(
                    f"step {step:5d}/{config.max_steps} | "
                    f"loss: {loss:.6f} | lr {lr:.4e} | norm: {norm:.4f} | "
                    f"dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.0f} | "
                    f"warming up..."
                )

            with open(log_file, "a") as f:
                f.write(f"{step} train {loss:.6f}\n")

    if ddp:
        destroy_process_group()

    if master_process:
        print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 model")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from (e.g., log/model_05000.pt)",
    )
    args = parser.parse_args()

    main(resume_from=args.resume)

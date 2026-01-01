import math
import os
import time

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from hellaswag import iterate_examples, render_example
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------------


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {"train", "val"}
        # tied to the structure in fineweb.py
        # get the shard filenames
        data_root = "edu_fineweb10B"
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
        # state
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
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


def get_device():
    result = "cpu"
    if torch.cuda.is_available():
        result = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        result = "mps"
    return result


def set_seed_on_device(device: str, seed: int):
    device_funct = {
        "cuda": torch.cuda.manual_seed_all,
        "mps": torch.mps.manual_seed,
        "cpu": torch.manual_seed,
    }[device]
    device_funct(seed)


def sync_on_device(device: str):
    device_funct = {
        "cuda": torch.cuda.synchronize,
        "mps": torch.mps.synchronize,
        "cpu": lambda: None,
    }[device]
    device_funct()


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# -----------------------------------------------------------------------------
# Training script
# -----------------------------------------------------------------------------
# consider running on rented gpu from https://cloud.lambda.ai/instances
# ddp mode is NVIDIA only but single can still be any device
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
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
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = get_device()
    print(f"using device: {device}")

torch.manual_seed(1337)
set_seed_on_device(device, 1337)

enc = tiktoken.get_encoding("gpt2")

# adjust batch size to fit in the gpu
# before memory optmization 16 seems to be around 43 gigs
# so fine on my macbook, but way to big for my 5070ti
# each training iteration is about 6-7 seconds after the first couple which are closer to 15
# around 2k tokens per second
# after moving to tensor float 32 I am still only seeing 2k tokens per second
#
# lets try B = 8 no tensor float uses 26gb and gets 4k tok/sec
# with tensor_float I get the same as without. It seems that apple doesn't suport this
#
# B=8 runs out of memory without memory optimization on 5070ti
# B=4 uses 10,288 MiB without optimization, around 18,000 tok/sec
# With both autocast and high precision we use just a little less 10,058 MiB but we get ~26,000 tok/sec
# B=6 we use 13,916 MiB
# Later still B=6 I got a crash
#
# Now after adding torch.compile(model) I am getting 48.5k tok/sec and using only 7,644 MiB vram (B=4)
# Move to B=8 and get 55.6k tok/sec and 13,198 MiB vram
#
# With flash attention we get 72k tok/sec and 8,534 MiB vram (B=8)
# With flash attention we get 74k tok/sec and 12,738 MiB vram (B=14)
#
# increase vocab_size to 50304 gets us to 78k tok/sec and 11,368 MiB MiB vram (B=14)
# 79k tok/sec and 12,768 MiB MiB vram (B=16)

total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
# karpathy used B = 64
B = 8  # micro batch size
T = 1024  # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, (
    "make sure total_batch_size is divisible by B * T * ddp_world_size"
)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)


torch.set_float32_matmul_precision("high")

# inline the forward pass and loss calculations
# recall that default vocab_size is 50257, but that isn't a power of 2
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# doens't seem to help on apple gpu
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_rank], output_device=ddp_rank)
raw_model = model.module if ddp else model

# implement variable learning rate
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
steps_completed = 0
total_time_elapsed = 0.0
avg_step_time = None
output_interval = 100
output_interval = 10
checkpoint_interval = 5000
checkpoint_interval = 10


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# Training hyperparameters
weight_decay = 0.1
learning_rate = 6e-4

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    device=device,
    master_process=master_process,
)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # once in a while evaluate our validation loss
    if step % output_interval == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # save the weights
            if step > 0 and (step % checkpoint_interval == 0 or last_step):
                # optionally write model checkpoints
                print("saving weights")
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if step % output_interval == 0 or last_step:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
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

    # once in a while generate from the model (except step 0, which is noise)
    if (step > 0 and step % output_interval == 0) or last_step:
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
                    logits, _ = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    # apple gpu doesn't seem to support this either I get no change in performance

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
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
        # Calculate time estimates
        steps_remaining = max_steps - steps_completed

        if avg_step_time is not None:
            estimated_time_remaining = steps_remaining * avg_step_time
            estimated_total_time = total_time_elapsed + estimated_time_remaining

            # Format time estimates
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                if hours > 0:
                    return f"{hours}h {minutes}m {secs}s"
                elif minutes > 0:
                    return f"{minutes}m {secs}s"
                else:
                    return f"{secs}s"

            elapsed_str = format_time(total_time_elapsed)
            remaining_str = format_time(estimated_time_remaining)
            total_str = format_time(estimated_total_time)

            progress_pct = (steps_completed / max_steps) * 100

            print(
                f"step {step:5d}/{max_steps} ({progress_pct:.1f}%) | "
                f"loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | "
                f"dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.0f} | "
                f"elapsed: {elapsed_str} | remaining: {remaining_str} | total: {total_str}"
            )
        else:
            # During initial steps, show simpler output
            print(
                f"step {step:5d}/{max_steps} | "
                f"loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | "
                f"dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.0f} | "
                f"warming up..."
            )
        # log the loss simply without time estimates
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

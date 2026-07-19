"""
src/training.py
================

A lightweight, dependency-minimal MLM pre-training loop. Supports mixed
precision (bf16 autocast on Ampere+ GPUs, fp16 autocast + ``GradScaler``
everywhere else) and tracks running training / validation MLM loss and
perplexity, designed to comfortably fit -- and run efficiently on -- a
single Colab T4.

Deliberately does *not* implement gradient accumulation, distributed
training, or checkpoint resumption: those add real complexity for a
benchmarking library whose whole point is a fast, legible baseline-vs-AttnRes
comparison. Bolt them on externally if you need them for a bigger run.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "fp16"  # one of {"fp16", "bf16", "no"}
    output_dir: str = "./checkpoints"


def _resolve_amp_dtype(mixed_precision: str, device: str):
    """Returns (autocast_dtype, use_grad_scaler). GradScaler is only needed
    for fp16 (bf16 has enough dynamic range that scaling is unnecessary,
    and fp32 has no under/overflow risk to begin with)."""
    if device != "cuda" or mixed_precision == "no":
        return torch.float32, False
    if mixed_precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("[training] bf16 requested but not supported on this GPU -- falling back to fp16.")
        else:
            return torch.bfloat16, False
    return torch.float16, True  # default: fp16 (T4-friendly, needs GradScaler)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with the standard BERT-style parameter split: no weight decay
    on biases or any 1-D parameter (LayerNorm weight/bias, RMSNorm has none,
    and crucially the AttentionResidualMix pseudo-query ``w``, itself a
    1-D vector, is correctly swept into the no-decay group by this rule)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-6)


def build_scheduler(optimizer: torch.optim.Optimizer, num_training_steps: int, warmup_ratio: float):
    """Linear warmup -> cosine decay to zero, the standard BERT-pretraining
    schedule shape."""
    num_warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / num_warmup_steps
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _perplexity(loss: float) -> float:
    # Clip before exponentiating: a bad/early checkpoint can have a very
    # large loss, and exp() of that overflows to `inf`, which is unhelpful
    # for logging (20 nats is already an astronomically bad perplexity).
    return math.exp(min(loss, 20.0))


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: str, amp_dtype: torch.dtype) -> Dict[str, float]:
    """Full pass over the validation set. Returns token-weighted average
    loss and the corresponding perplexity (exp(loss))."""
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch in val_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            out = model(**batch)

        n_tokens = int((batch["labels"] != -100).sum().item())
        n_tokens = max(n_tokens, 1)  # guard against an all-unmasked batch
        total_loss += out["loss"].item() * n_tokens
        total_tokens += n_tokens

    model.train()
    avg_loss = total_loss / total_tokens
    return {"val_loss": avg_loss, "val_perplexity": _perplexity(avg_loss)}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainerConfig,
    run_name: str = "run",
) -> Dict[str, List[float]]:
    """Main training loop.

    Returns a ``history`` dict of logged metrics over time (steps, running
    train loss/perplexity, learning rate, and periodic val loss/perplexity)
    -- handy for plotting the AttnRes vs. baseline curves against each
    other after the fact.
    """
    device = config.device
    model.to(device)
    model.train()

    optimizer = build_optimizer(model, config.lr, config.weight_decay)
    num_training_steps = config.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, num_training_steps, config.warmup_ratio)

    amp_dtype, use_scaler = _resolve_amp_dtype(config.mixed_precision, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    history: Dict[str, List[float]] = {
        "step": [], "train_loss": [], "train_ppl": [], "lr": [],
        "val_step": [], "val_loss": [], "val_ppl": [],
    }

    running_loss, running_tokens = 0.0, 0
    global_step = 0
    start_time = time.time()

    print(f"[{run_name}] device={device} amp_dtype={amp_dtype} steps/epoch={len(train_loader)} "
          f"total_steps={num_training_steps}")

    for epoch in range(config.epochs):
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                out = model(**batch)
                loss = out["loss"]

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # required before clipping under GradScaler
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            scheduler.step()

            n_tokens = max(int((batch["labels"] != -100).sum().item()), 1)
            running_loss += loss.item() * n_tokens
            running_tokens += n_tokens
            global_step += 1

            if global_step % config.log_every == 0:
                avg_loss = running_loss / running_tokens
                elapsed = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[{run_name}] epoch {epoch} step {global_step}/{num_training_steps} "
                    f"| loss {avg_loss:.4f} | ppl {_perplexity(avg_loss):.2f} "
                    f"| lr {current_lr:.2e} | {elapsed:.1f}s elapsed"
                )
                history["step"].append(global_step)
                history["train_loss"].append(avg_loss)
                history["train_ppl"].append(_perplexity(avg_loss))
                history["lr"].append(current_lr)
                running_loss, running_tokens = 0.0, 0

            if global_step % config.eval_every == 0:
                metrics = evaluate(model, val_loader, device, amp_dtype)
                print(f"[{run_name}]   >> val_loss {metrics['val_loss']:.4f} "
                      f"| val_ppl {metrics['val_perplexity']:.2f}")
                history["val_step"].append(global_step)
                history["val_loss"].append(metrics["val_loss"])
                history["val_ppl"].append(metrics["val_perplexity"])

    # Final full validation pass at the very end of training.
    final_metrics = evaluate(model, val_loader, device, amp_dtype)
    print(f"[{run_name}] FINAL >> val_loss {final_metrics['val_loss']:.4f} "
          f"| val_ppl {final_metrics['val_perplexity']:.2f}")
    history["val_step"].append(global_step)
    history["val_loss"].append(final_metrics["val_loss"])
    history["val_ppl"].append(final_metrics["val_perplexity"])

    return history

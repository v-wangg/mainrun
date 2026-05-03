import inspect
import math

import torch
import torch.nn as nn

from config import Hyperparameters

def configure_optimizer(model: nn.Module, args: Hyperparameters, device: str):
    # Any param with dim() >= 2 gets weight decay (Linear weights,
    # token_emb.weight, pos_emb), everything else (biases, LayerNorm params) does not.
    # head.weight is tied to token_emb.weight; named_parameters() deduplicates.
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optim_groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay_params = sum(p.numel() for p in decay)
    n_no_decay_params = sum(p.numel() for p in no_decay)
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"
    opt = torch.optim.AdamW(
        optim_groups,
        lr=args.lr,
        betas=args.adam_betas,
        eps=args.adam_eps,
        fused=use_fused,
    )
    return opt, n_decay_params, n_no_decay_params, len(decay), len(no_decay), use_fused

def get_lr(step: int, peak_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)

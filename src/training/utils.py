"""
Training utilities: EMA scheduler, learning rate scheduler, logging helpers.
"""

import math
from typing import Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR


class EMAScheduler:
    """
    Cosine EMA momentum schedule.

    momentum = ema_end - (ema_end - ema_start) * (cos(pi * step / total_steps) + 1) / 2

    At step 0: momentum ≈ ema_start
    At step total_steps: momentum = ema_end
    """

    def __init__(
        self,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        total_steps: int = 1000,
    ):
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.total_steps = total_steps
        self.current_step = 0

    def update(self):
        """Advance one step and return current momentum."""
        self.current_step += 1
        return self.get_momentum()

    def get_momentum(self) -> float:
        """Get current EMA momentum."""
        if self.current_step >= self.total_steps:
            return self.ema_end

        progress = self.current_step / self.total_steps
        cosine = math.cos(math.pi * progress)
        momentum = self.ema_end - (self.ema_end - self.ema_start) * (cosine + 1.0) / 2.0
        return momentum


class CosineLRScheduler:
    """
    Cosine learning rate schedule with linear warmup.

    During warmup: lr = base_lr * step / warmup_steps
    After warmup: lr = min_lr + (base_lr - min_lr) * (cos(pi * (step - warmup) / (total - warmup)) + 1) / 2
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> float:
        """Advance one step and return current LR."""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / max(1, self.warmup_steps)

        if self.current_step >= self.total_steps:
            return self.min_lr

        # Cosine decay
        decay_steps = self.total_steps - self.warmup_steps
        progress = (self.current_step - self.warmup_steps) / decay_steps
        cosine = math.cos(math.pi * progress)
        lr = self.min_lr + (self.base_lr - self.min_lr) * (cosine + 1.0) / 2.0
        return lr


def get_total_steps(
    num_samples: int,
    batch_size: int,
    epochs: int,
) -> int:
    """Calculate total training steps."""
    steps_per_epoch = max(1, num_samples // batch_size)
    return steps_per_epoch * epochs


def format_metrics(metrics: dict) -> str:
    """Format a metrics dict for logging."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)
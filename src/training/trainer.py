"""
I-JEPA training loop.

Manages the full training workflow:
    - Mask generation per batch
    - Forward pass (context → target → predict → loss)
    - Backward pass (context encoder + predictor only)
    - EMA update of target encoder
    - Logging and checkpointing
"""

import os
import math
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..models.ijepa import IJPEModel
from ..data.masks import generate_mask_batch
from .utils import CosineLRScheduler, format_metrics


class IJEPATrainer:
    """
    Trainer for I-JEPA pretraining.

    Args:
        model: I-JEPA model instance.
        train_loader: DataLoader for pretraining images.
        cfg: Training configuration dict.
    """

    def __init__(
        self,
        model: IJPEModel,
        train_loader: DataLoader,
        cfg: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.cfg = cfg

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer — only context encoder and predictor
        params = list(self.model.context_encoder.parameters()) + \
                 list(self.model.predictor.parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg["training"]["lr"],
            betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
            weight_decay=cfg["training"]["weight_decay"],
        )

        # AMP scaler
        self.use_amp = cfg["training"].get("use_amp", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Learning rate scheduler
        total_steps = len(train_loader) * cfg["training"]["epochs"]
        warmup_steps = cfg["training"]["warmup_epochs"] * len(train_loader)
        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            base_lr=cfg["training"]["lr"],
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=cfg["training"]["min_lr"],
        )

        # Mask parameters
        self.mask_cfg = cfg.get("mask", {})
        self.num_context_blocks = self.mask_cfg.get("num_context_blocks", 4)
        self.num_target_blocks = self.mask_cfg.get("num_target_blocks", 4)
        self.block_scale = tuple(self.mask_cfg.get("block_scale", [0.15, 0.2]))
        self.aspect_ratio = tuple(self.mask_cfg.get("aspect_ratio", [0.75, 1.5]))
        self.overlap_allowed = self.mask_cfg.get("overlap_allowed", False)

        self.img_size = cfg["data"]["image_size"]
        self.patch_size = cfg["model"]["patch_size"]

        # Gradient clipping
        self.gradient_clip = cfg["training"].get("gradient_clip", None)

        # Logging
        self.log_dir = Path(cfg["paths"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.log_every = cfg["training"]["log_every"]

        # Checkpointing
        self.checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = cfg["training"]["save_every"]

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # Resume from checkpoint if specified
        resume = cfg["paths"].get("resume_from")
        if resume and resume != "null" and resume is not None:
            self._resume(resume)

        print(f"Trainer initialized on {self.device}")
        print(f"  Use AMP: {self.use_amp}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch.

        Returns dict of average metrics for the epoch.
        """
        self.model.train()

        total_loss = 0.0
        n_batches = len(self.train_loader)
        epoch_start = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            batch_size = images.shape[0]

            # Generate masks for this batch
            context_mask, target_masks = generate_mask_batch(
                batch_size=batch_size,
                image_size=self.img_size,
                patch_size=self.patch_size,
                num_context_blocks=self.num_context_blocks,
                num_target_blocks=self.num_target_blocks,
                block_scale_range=self.block_scale,
                aspect_ratio_range=self.aspect_ratio,
                overlap_allowed=self.overlap_allowed,
            )

            batch_loss = 0.0

            # Process each target block independently
            for t in range(self.num_target_blocks):
                tgt_mask = target_masks[t].to(self.device)
                ctx_mask = context_mask.to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        _, _, loss = self.model(images, ctx_mask, tgt_mask)
                    self.scaler.scale(loss).backward()
                else:
                    _, _, loss = self.model(images, ctx_mask, tgt_mask)
                    loss.backward()

                batch_loss += loss.item()

            # Average loss over target blocks
            batch_loss /= self.num_target_blocks

            # Gradient clipping
            if self.use_amp:
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()

            self.optimizer.zero_grad()

            # EMA update target encoder
            ema_momentum = self._get_ema_momentum()
            self.model.ema_update(ema_momentum)

            # LR step
            lr = self.lr_scheduler.step()

            # Logging
            total_loss += batch_loss

            if self.global_step % self.log_every == 0:
                metrics = {
                    "loss": batch_loss,
                    "lr": lr,
                    "ema_momentum": ema_momentum,
                }
                self.writer.add_scalar("train/loss", batch_loss, self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)
                self.writer.add_scalar("train/ema", ema_momentum, self.global_step)
                pbar.set_postfix(metrics)

            self.global_step += 1

        epoch_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        return {
            "loss": epoch_loss,
            "lr": lr,
            "ema": ema_momentum,
            "time": epoch_time,
        }

    def _get_ema_momentum(self) -> float:
        """
        Get current EMA momentum using a cosine schedule.

        momentum goes from ema_start → ema_end over total steps.
        """
        ema_start = self.mask_cfg.get("ema_start", 0.996)
        ema_end = self.mask_cfg.get("ema_end", 1.0)
        total_steps = len(self.train_loader) * self.cfg["training"]["epochs"]

        if self.global_step >= total_steps:
            return ema_end

        progress = self.global_step / total_steps
        cosine = math.cos(math.pi * progress)
        momentum = ema_end - (ema_end - ema_start) * (cosine + 1.0) / 2.0
        return momentum

    def train(self):
        """Run full pretraining loop."""
        total_epochs = self.cfg["training"]["epochs"]
        print(f"\nStarting I-JEPA pretraining for {total_epochs} epochs")
        print(f"  Logs: {self.log_dir}")
        print(f"  Checkpoints: {self.checkpoint_dir}")

        for epoch in range(self.current_epoch + 1, total_epochs + 1):
            self.current_epoch = epoch
            metrics = self.train_epoch(epoch)

            print(
                f"\nEpoch {epoch}/{total_epochs} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"LR: {metrics['lr']:.6f} | "
                f"EMA: {metrics['ema']:.3f} | "
                f"Time: {metrics['time']:.1f}s"
            )

            # Save checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(f"ijepa_epoch_{epoch}.pth", metrics)

            # Save best
            if metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                self._save_checkpoint("ijepa_best.pth", metrics)
                print(f"  → Best model saved (loss: {self.best_loss:.4f})")

        # Final save
        self._save_checkpoint("ijepa_final.pth", metrics)
        self.writer.close()
        print("\nPretraining complete.")

    def _save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "context_encoder_state_dict": self.model.context_encoder.state_dict(),
            "target_encoder_state_dict": self.model.target_encoder.state_dict(),
            "predictor_state_dict": self.model.predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "cfg": self.cfg,
        }

        torch.save(checkpoint, str(path))
        print(f"  Checkpoint saved: {path}")

    def _resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        path = Path(os.path.expanduser(checkpoint_path))
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("metrics", {}).get("loss", float("inf"))

        print(f"Resumed from {path} (epoch {self.current_epoch}, step {self.global_step})")
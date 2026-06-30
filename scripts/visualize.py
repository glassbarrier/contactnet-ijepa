#!/usr/bin/env python3
"""
Visualization script for I-JEPA anomaly detection project.

Generates figures suitable for reports/papers from existing training and
evaluation artifacts. Reads TensorBoard event files, checkpoints, and
anomaly detection results — no changes to existing code required.

Usage:
    # Training curves (loss, lr, ema) from TensorBoard logs
    python scripts/visualize.py --mode training --log_dir logs/

    # I-JEPA mask visualization
    python scripts/visualize.py --mode masks --config config/ijepa_pretrain.yaml

    # AUROC bar chart from anomaly detection results
    python scripts/visualize.py --mode anomaly --results results/anomaly_results.json

    # Anomaly heatmaps on test images
    python scripts/visualize.py --mode heatmaps --config config/anomaly_detection.yaml \\
        -o features.checkpoint=checkpoints/ijepa_best.pth

    # Generate all figures
    python scripts/visualize.py --mode all --config config/anomaly_detection.yaml \\
        --log_dir logs/ --results results/anomaly_results.json \\
        -o features.checkpoint=checkpoints/ijepa_best.pth

Output: figures saved to ./figures/ by default (override with --output_dir).

TensorBoard dependency note:
    If `tensorboard` is not installed, training curves will instead read per-epoch
    metrics from checkpoint files in the checkpoints/ directory.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Project imports (available when run from repo root)
from src.data.masks import generate_block_masks
from src.data.dataset import AnomalyDataset, _discover_categories
from src.data.augmentations import IJEPAAugmentations
from src.models.vision_transformer import VisionTransformer
from src.features.extract import FeatureExtractor
from src.features.bank import FeatureBank
from src.anomaly.padim import PaDiMDetector
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Style defaults — consistent look across all figures
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ===========================================================================
#  Mode 1 — Training curves
# ===========================================================================

def _read_tensorboard_logs(log_dir: str) -> Optional[Dict[str, List[float]]]:
    """Parse TensorBoard event files. Returns None if tensorboard unavailable."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None

    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"[WARN] Log directory not found: {log_dir}")
        return None

    event_files = list(log_path.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"[WARN] No TensorBoard event files found in {log_dir}")
        return None

    ea = EventAccumulator(str(log_path))
    ea.Reload()

    scalars = ea.Tags().get("scalars", [])
    result: Dict[str, List[float]] = defaultdict(list)

    for tag in scalars:
        events = ea.Scalars(tag)
        # Each event has .step and .value
        for ev in events:
            result[tag].append((ev.step, ev.value))

    # Sort by step
    for tag in result:
        result[tag].sort(key=lambda x: x[0])

    return dict(result)


def _read_checkpoint_metrics(checkpoint_dir: str) -> Optional[Dict[str, list]]:
    """Fallback: gather per-epoch metrics from checkpoint files."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return None

    pth_files = sorted(ckpt_path.glob("ijepa_epoch_*.pth"))
    if not pth_files:
        # Try best/final checkpoints only
        pth_files = sorted(ckpt_path.glob("ijepa_*.pth"))

    if not pth_files:
        return None

    epochs, losses, lrs, emas = [], [], [], []
    for p in pth_files:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        m = ckpt.get("metrics", {})
        epochs.append(ckpt.get("epoch", len(epochs)))
        losses.append(m.get("loss", float("nan")))
        lrs.append(m.get("lr", float("nan")))
        emas.append(m.get("ema", float("nan")))

    # Sort by epoch
    order = np.argsort(epochs)
    return {
        "train/loss": [(int(epochs[i]), losses[i]) for i in order],
        "train/lr":   [(int(epochs[i]), lrs[i])   for i in order],
        "train/ema":  [(int(epochs[i]), emas[i])  for i in order],
    }


def plot_training(
    log_dir: str = "logs/",
    checkpoint_dir: str = "checkpoints/",
    output_dir: str = "figures/",
    figsize: Tuple[int, int] = (14, 4),
):
    """Generate training curve plots: loss, learning rate, EMA momentum."""
    data = _read_tensorboard_logs(log_dir)

    if data is None or not data:
        print("TensorBoard logs unavailable; falling back to checkpoint metrics.")
        data = _read_checkpoint_metrics(checkpoint_dir)

    if data is None or not data:
        print("[ERROR] No training metrics found. Run pretraining first.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    def _plot_one(ax, tag: str, ylabel: str, title: str, color: str = "#1f77b4"):
        if tag not in data:
            ax.set_title(f"{title}\n(no data)")
            return
        steps, vals = zip(*data[tag])
        ax.plot(steps, vals, linewidth=0.8, color=color, alpha=0.9)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(6))

    _plot_one(axes[0], "train/loss", "Loss", "Training Loss", "#d62728")
    _plot_one(axes[1], "train/lr", "Learning Rate", "Learning Rate Schedule", "#2ca02c")
    _plot_one(axes[2], "train/ema", "EMA Momentum", "Target Encoder EMA", "#9467bd")

    plt.tight_layout()
    filepath = out / "training_curves.png"
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved {filepath}")


# ===========================================================================
#  Mode 2 — I-JEPA mask visualization
# ===========================================================================

def _draw_mask_grid(ax, mask_1d: torch.Tensor, patch_size: int,
                     image_size: int, color, alpha: float = 0.45):
    """Draw filled rectangles for each True patch in a 1-D boolean mask."""
    num_per_side = image_size // patch_size
    mask_2d = mask_1d.reshape(num_per_side, num_per_side)
    for i in range(num_per_side):
        for j in range(num_per_side):
            if mask_2d[i, j]:
                rect = plt.Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size, patch_size,
                    facecolor=color, edgecolor="none", alpha=alpha,
                )
                ax.add_patch(rect)


def plot_masks(
    config_path: str = "config/ijepa_pretrain.yaml",
    output_dir: str = "figures/",
    num_samples: int = 4,
    data_root: Optional[str] = None,
    seed: int = 42,
):
    """Visualize I-JEPA context / target block masks on sample images."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = load_config(config_path)
    img_size = cfg["data"]["image_size"]
    patch_size = cfg["model"]["patch_size"]
    mc = cfg.get("mask", {})

    num_context = mc.get("num_context_blocks", 4)
    num_target = mc.get("num_target_blocks", 4)
    block_scale = tuple(mc.get("block_scale", [0.15, 0.2]))
    aspect_ratio = tuple(mc.get("aspect_ratio", [0.75, 1.5]))

    # Try loading a few real images
    root = data_root or os.path.expanduser(cfg["data"]["root"])
    categories = _discover_categories(root)
    if not categories:
        print("[ERROR] No categories found — set data.root correctly.")
        return

    # Pick a category and load some images
    import torchvision.transforms as T
    from PIL import Image

    sample_images = []
    cat = categories[0]
    train_good = Path(root) / cat / "train" / "good"
    if not train_good.exists():
        print(f"[WARN] {train_good} not found; skipping real images.")
    else:
        files = sorted(train_good.glob("*.png")) + sorted(train_good.glob("*.jpg"))
        chosen = np.random.choice(min(len(files), num_samples), num_samples, replace=False)
        for idx in chosen:
            img = Image.open(files[idx]).convert("RGB")
            img = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])(img)
            sample_images.append(img)

    # If no real images, use a simple color gradient placeholder
    if not sample_images:
        print("No real images found; using synthetic placeholders.")
        for i in range(num_samples):
            g = torch.linspace(0, 1, img_size)
            synthetic = torch.stack([
                g.view(1, -1).expand(img_size, img_size) * 0.7,
                g.view(-1, 1).expand(img_size, img_size) * 0.5,
                torch.full((img_size, img_size), 0.3),
            ], dim=0)
            sample_images.append(synthetic)

    num_per_side = img_size // patch_size
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Context block colors, target block colors
    ctx_colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    tgt_colors = ["#e91e63", "#00bcd4", "#ff9800", "#673ab7"]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3.5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for s in range(num_samples):
        img_tensor = sample_images[s]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        ctx_mask, tgt_masks = generate_block_masks(
            image_size=img_size, patch_size=patch_size,
            num_context_blocks=num_context, num_target_blocks=num_target,
            block_scale_range=block_scale, aspect_ratio_range=aspect_ratio,
            overlap_allowed=False,
        )

        # Col 1: Context mask overlay
        axes[s, 0].imshow(img_np)
        _draw_mask_grid(axes[s, 0], ctx_mask, patch_size, img_size,
                         color=ctx_colors[0], alpha=0.35)
        axes[s, 0].set_title(f"Context blocks ({num_context})")
        axes[s, 0].axis("off")

        # Col 2: Target mask overlay (all 4 blocks)
        axes[s, 1].imshow(img_np)
        combined_tgt = torch.zeros_like(ctx_mask, dtype=torch.bool)
        for t in range(num_target):
            _draw_mask_grid(axes[s, 1], tgt_masks[t], patch_size, img_size,
                             color=tgt_colors[t], alpha=0.35)
            combined_tgt = combined_tgt | tgt_masks[t]
        axes[s, 1].set_title(f"Target blocks ({num_target})")
        axes[s, 1].axis("off")

        # Col 3: Overlay only — context visible, target hidden (model input)
        axes[s, 2].imshow(np.ones_like(img_np) * 0.15)  # dark bg
        _draw_mask_grid(axes[s, 2], ctx_mask & ~combined_tgt, patch_size, img_size,
                         color=ctx_colors[0], alpha=0.35)
        axes[s, 2].set_title("Context / Target split")
        axes[s, 2].axis("off")

    plt.tight_layout()
    filepath = out / "ijepa_masks.png"
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved {filepath}")


# ===========================================================================
#  Mode 3 — Anomaly detection AUROC bar chart
# ===========================================================================

def plot_anomaly_results(
    results_path: str = "results/anomaly_results.json",
    output_dir: str = "figures/",
    figsize: Tuple[int, int] = (14, 5),
):
    """Generate per-category AUROC bar chart from anomaly_results.json."""
    rp = Path(results_path)
    if not rp.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        print("  Run anomaly_detect.py first to generate results.")
        return

    with open(rp) as f:
        results = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Sort categories by image AUROC descending
    sorted_cats = sorted(results.items(),
                         key=lambda kv: kv[1].get("image_auroc", 0),
                         reverse=True)
    names = [c[0] for c in sorted_cats]
    img_aurocs = [c[1].get("image_auroc", 0) for c in sorted_cats]
    pixel_aurocs = [c[1].get("pixel_auroc", 0) for c in sorted_cats]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width / 2, img_aurocs, width, label="Image AUROC",
                    color="#3498db", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, pixel_aurocs, width, label="Pixel AUROC",
                    color="#e74c3c", edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("AUROC")
    ax.set_title("Anomaly Detection Performance by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # Add mean line
    mean_img = np.mean(img_aurocs)
    mean_pix = np.mean(pixel_aurocs)
    ax.axhline(y=mean_img, color="#3498db", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=mean_pix, color="#e74c3c", linestyle="--", alpha=0.4, linewidth=1)

    plt.tight_layout()
    filepath = out / "anomaly_auroc.png"
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved {filepath}")

    # Also print numerical summary
    print(f"\n  Mean Image AUROC: {mean_img:.4f}")
    print(f"  Mean Pixel AUROC: {mean_pix:.4f}")


# ===========================================================================
#  Mode 4 — Anomaly heatmaps
# ===========================================================================

def plot_heatmaps(
    config_path: str = "config/anomaly_detection.yaml",
    output_dir: str = "figures/",
    checkpoint_override: Optional[str] = None,
    num_normal: int = 3,
    num_anomaly: int = 3,
    seed: int = 42,
):
    """
    Generate anomaly heatmap overlaid on test images.

    For each category, selects the worst-scoring normal and anomaly images,
    then overlays the anomaly score map (upsampled to image resolution).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = load_config(config_path)
    data_root = os.path.expanduser(cfg["data"]["root"])
    img_size = cfg["data"]["image_size"]

    checkpoint_path = checkpoint_override or cfg["features"]["checkpoint"]
    checkpoint_path = os.path.expanduser(checkpoint_path)
    layer_indices = cfg["features"].get("layers", [4, 6, 8, 10, 12])
    reduce_dim = cfg["features"].get("reduce_dim", None)

    categories = cfg["evaluation"].get("categories", [])
    if not categories:
        categories = _discover_categories(data_root)

    # Load checkpoint + model
    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("cfg", {})
    embed_dim = model_cfg.get("model", {}).get("embed_dim", 384)
    depth = model_cfg.get("model", {}).get("depth", 12)
    heads = model_cfg.get("model", {}).get("heads", 6)
    patch_size = model_cfg.get("model", {}).get("patch_size", 16)

    encoder = VisionTransformer(
        img_size=img_size, patch_size=patch_size,
        embed_dim=embed_dim, depth=depth, num_heads=heads,
    )
    ctx_state = ckpt.get("context_encoder_state_dict", {})
    if not ctx_state:
        fallback = ckpt.get("model_state_dict", {})
        ctx_state = {k.replace("context_encoder.", ""): v
                     for k, v in fallback.items()
                     if k.startswith("context_encoder.")}
    encoder.load_state_dict(ctx_state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor(
        encoder=encoder, layer_indices=layer_indices,
        patch_size=patch_size, img_size=img_size, device=device,
    )

    bank = FeatureBank(extractor=extractor, categories=categories,
                       reduce_dim=reduce_dim, device=device)

    bank_dir = cfg["evaluation"].get("output_dir", "results/")
    bank_path = os.path.join(bank_dir, "feature_bank.pkl")
    if os.path.exists(bank_path):
        bank.load(bank_dir)
    else:
        bank.build(root=data_root, image_size=img_size,
                   batch_size=cfg["data"]["batch_size"],
                   num_workers=cfg["data"]["num_workers"])

    detector = PaDiMDetector(
        extractor=extractor, means=bank.means, cov_invs=bank.cov_invs,
        proj_matrices=bank.proj_matrices,
        top_k_ratio=cfg["anomaly"].get("top_k", 0.01),
    )

    augmentations = IJEPAAugmentations(image_size=img_size)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import cv2
    from matplotlib.colors import LinearSegmentedColormap

    # Custom heatmap colormap: transparent → yellow → red
    heat_cmap = LinearSegmentedColormap.from_list(
        "heat", [(0, 0, 0, 0), (1, 1, 0, 0.6), (1, 0, 0, 0.85)]
    )

    # Pick a subset of categories for display (max 6)
    display_cats = categories[:6]
    num_cols = num_normal + num_anomaly

    for cat in display_cats:
        if cat not in bank.means:
            print(f"  Skipping {cat} — not in feature bank")
            continue

        test_dataset = AnomalyDataset(
            root=data_root, category=cat, split="test",
            transform=augmentations.get_eval(),
        )

        if len(test_dataset) == 0:
            continue

        # Score all test images
        scores, labels, maps, images_t = [], [], [], []
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            img_t = sample["image"]
            amap, iscore = detector.score_image(img_t.unsqueeze(0), cat)
            scores.append(iscore)
            labels.append(sample["label"])
            maps.append(amap)
            images_t.append(img_t)

        scores = np.array(scores)
        labels = np.array(labels)

        norm_idx = np.where(labels == 0)[0]
        anom_idx = np.where(labels == 1)[0]

        if len(norm_idx) == 0 or len(anom_idx) == 0:
            print(f"  Skipping {cat} — need both normal & anomaly test images")
            continue

        # Select top-scoring (most anomalous-like) normals, worst anomalies
        top_norm = norm_idx[np.argsort(scores[norm_idx])[-num_normal:]]
        top_anom = anom_idx[np.argsort(scores[anom_idx])[-num_anomaly:]]

        fig, axes = plt.subplots(2, num_cols, figsize=(3 * num_cols, 7))

        # Row 1: Normal images
        for j, idx in enumerate(top_norm):
            img_np = images_t[idx].permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            amap = maps[idx]
            amap_up = cv2.resize(amap, (img_size, img_size),
                                 interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 1] for display
            amap_up = (amap_up - amap_up.min()) / (amap_up.max() - amap_up.min() + 1e-8)

            axes[0, j].imshow(img_np)
            axes[0, j].imshow(amap_up, cmap=heat_cmap, alpha=0.7)
            axes[0, j].set_title(f"Normal (score={scores[idx]:.1f})", fontsize=9)
            axes[0, j].axis("off")

        # Row 2: Anomaly images
        for j, idx in enumerate(top_anom):
            img_np = images_t[idx].permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            amap = maps[idx]
            amap_up = cv2.resize(amap, (img_size, img_size),
                                 interpolation=cv2.INTER_LINEAR)
            amap_up = (amap_up - amap_up.min()) / (amap_up.max() - amap_up.min() + 1e-8)

            axes[1, j].imshow(img_np)
            axes[1, j].imshow(amap_up, cmap=heat_cmap, alpha=0.7)
            axes[1, j].set_title(f"Anomaly (score={scores[idx]:.1f})", fontsize=9)
            axes[1, j].axis("off")

        axes[0, 0].set_ylabel("Normal", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Anomaly", fontsize=12, fontweight="bold")
        fig.suptitle(f"Anomaly Heatmaps — {cat}",
                      fontsize=14, fontweight="bold", y=1.01)

        plt.tight_layout()
        filepath = out / f"heatmaps_{cat}.png"
        fig.savefig(filepath)
        plt.close(fig)
        print(f"  Saved {filepath}")


# ===========================================================================
#  Main CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="I-JEPA Visualization — generate figures for reports/papers",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["training", "masks", "anomaly", "heatmaps", "all"],
        help="Which visualization to generate",
    )
    parser.add_argument(
        "--config", default="config/anomaly_detection.yaml",
        help="Path to YAML config (for masks/heatmaps modes)",
    )
    parser.add_argument(
        "--log_dir", default="logs/",
        help="TensorBoard log directory (for training mode)",
    )
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints/",
        help="Checkpoint directory fallback (for training mode)",
    )
    parser.add_argument(
        "--results", default="results/anomaly_results.json",
        help="Path to anomaly_results.json (for anomaly mode)",
    )
    parser.add_argument(
        "--output_dir", default="figures/",
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "-o", "--override", nargs="*", default=[],
        help="CLI overrides in key=value format (e.g. features.checkpoint=...)",
    )

    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build checkpoint override string for heatmaps mode
    ckpt_override = None
    for ov in args.override:
        if ov.startswith("features.checkpoint="):
            ckpt_override = ov.split("=", 1)[1]

    modes = (
        ["training", "masks", "anomaly", "heatmaps"] if args.mode == "all"
        else [args.mode]
    )

    for mode in modes:
        print(f"\n{'=' * 50}")
        print(f"  Generating: {mode}")
        print(f"{'=' * 50}")

        if mode == "training":
            plot_training(
                log_dir=args.log_dir,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir,
            )
        elif mode == "masks":
            plot_masks(
                config_path=args.config,
                output_dir=args.output_dir,
            )
        elif mode == "anomaly":
            plot_anomaly_results(
                results_path=args.results,
                output_dir=args.output_dir,
            )
        elif mode == "heatmaps":
            plot_heatmaps(
                config_path=args.config,
                output_dir=args.output_dir,
                checkpoint_override=ckpt_override,
            )

    print(f"\nAll figures saved to {out.resolve()}/")


if __name__ == "__main__":
    main()
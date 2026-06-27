#!/usr/bin/env python3
"""
I-JEPA Pretraining Entry Point.

Usage:
    python scripts/pretrain.py --config config/ijepa_pretrain.yaml
    python scripts/pretrain.py --config config/ijepa_pretrain.yaml \
        -o data.batch_size=64 training.epochs=100 data.root=/path/to/data
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, create_arg_parser
from src.data.dataset import PretrainDataset
from src.data.augmentations import IJEPAAugmentations
from src.models.ijepa import IJPEModel
from src.training.trainer import IJEPATrainer

from torch.utils.data import DataLoader


def main():
    parser = create_arg_parser("I-JEPA Pretraining for Contact Network Components")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config, cli_overrides=args.override)
    print(f"Loaded config: {args.config}")

    # Data augmentations
    augmentations = IJEPAAugmentations(
        image_size=cfg["data"]["image_size"],
    )

    # Dataset
    dataset = PretrainDataset(
        root=cfg["data"]["root"],
        categories=cfg["data"].get("categories", []) or None,
        transform=augmentations.get_train(),
    )

    # DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    # Model
    model = IJPEModel(
        img_size=cfg["data"]["image_size"],
        patch_size=cfg["model"]["patch_size"],
        context_embed_dim=cfg["model"]["embed_dim"],
        context_depth=cfg["model"]["depth"],
        context_heads=cfg["model"]["heads"],
        predictor_embed_dim=cfg["model"]["predictor"]["embed_dim"],
        predictor_depth=cfg["model"]["predictor"]["depth"],
        predictor_heads=cfg["model"]["predictor"]["heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
    )

    # Count parameters
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Context encoder: {count_params(model.context_encoder):,}")
    print(f"  Target encoder:  {count_params(model.target_encoder):,} (frozen)")
    print(f"  Predictor:       {count_params(model.predictor):,}")
    print(f"  Total trainable: {count_params(model):,}")

    # Trainer
    trainer = IJEPATrainer(
        model=model,
        train_loader=data_loader,
        cfg=cfg,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
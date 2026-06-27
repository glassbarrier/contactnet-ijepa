#!/usr/bin/env python3
"""
Feature extraction from pretrained I-JEPA model.

Loads a pretrained checkpoint, extracts multi-scale patch features from all
normal training images, and saves per-category feature banks for anomaly detection.

Usage:
    python scripts/extract_features.py --config config/anomaly_detection.yaml \
        -o features.checkpoint=checkpoints/ijepa_best.pth
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_config, create_arg_parser
from src.data.dataset import _discover_categories
from src.models.vision_transformer import VisionTransformer
from src.features.extract import FeatureExtractor
from src.features.bank import FeatureBank


def main():
    parser = create_arg_parser("I-JEPA Feature Extraction")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.override)
    print(f"Loaded config: {args.config}")

    # Parse config
    data_root = os.path.expanduser(cfg["data"]["root"])
    img_size = cfg["data"]["image_size"]

    checkpoint_path = os.path.expanduser(cfg["features"]["checkpoint"])
    layer_indices = cfg["features"].get("layers", [4, 6, 8, 10, 12])
    reduce_dim = cfg["features"].get("reduce_dim", None)

    categories = cfg["evaluation"].get("categories", [])
    if not categories:
        categories = _discover_categories(data_root)

    print(f"Categories: {categories}")
    print(f"Feature layers: {layer_indices}")

    # Determine model architecture from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("cfg", {})

    embed_dim = model_cfg.get("model", {}).get("embed_dim", 384)
    depth = model_cfg.get("model", {}).get("depth", 12)
    heads = model_cfg.get("model", {}).get("heads", 6)
    patch_size = model_cfg.get("model", {}).get("patch_size", 16)

    # Reconstruct context encoder
    encoder = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=heads,
    )

    # Load context encoder weights
    context_state = checkpoint.get("context_encoder_state_dict", {})
    if not context_state:
        # Fallback: try loading full model then extract
        fallback = checkpoint.get("model_state_dict", {})
        context_state = {
            k.replace("context_encoder.", ""): v
            for k, v in fallback.items()
            if k.startswith("context_encoder.")
        }

    encoder.load_state_dict(context_state, strict=False)
    print(f"Loaded context encoder from {checkpoint_path}")

    # Feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor(
        encoder=encoder,
        layer_indices=layer_indices,
        patch_size=patch_size,
        img_size=img_size,
        device=device,
    )

    # Feature bank
    bank = FeatureBank(
        extractor=extractor,
        categories=categories,
        reduce_dim=reduce_dim,
    )

    # Build banks
    bank.build(
        root=data_root,
        image_size=img_size,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # Save
    output_dir = cfg["evaluation"].get("output_dir", "feature_banks/")
    bank.save(output_dir)


if __name__ == "__main__":
    main()
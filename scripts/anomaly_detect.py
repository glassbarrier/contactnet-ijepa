#!/usr/bin/env python3
"""
Anomaly Detection Script.

Runs anomaly detection evaluation on all contact network component categories
using pretrained I-JEPA features and PaDiM-style patch distribution modeling.

Usage:
    python scripts/anomaly_detect.py --config config/anomaly_detection.yaml
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
from src.anomaly.padim import PaDiMDetector
from src.anomaly.detector import AnomalyEvaluator


def main():
    parser = create_arg_parser("I-JEPA Anomaly Detection Evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=args.override)
    print(f"Loaded config: {args.config}")

    data_root = os.path.expanduser(cfg["data"]["root"])
    img_size = cfg["data"]["image_size"]
    checkpoint_path = os.path.expanduser(cfg["features"]["checkpoint"])
    layer_indices = cfg["features"].get("layers", [4, 6, 8, 10, 12])
    reduce_dim = cfg["features"].get("reduce_dim", None)

    categories = cfg["evaluation"].get("categories", [])
    if not categories:
        categories = _discover_categories(data_root)

    print(f"Categories: {categories}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint to get model architecture
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

    # Load weights
    context_state = checkpoint.get("context_encoder_state_dict", {})
    if not context_state:
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

    # Try loading pre-built feature bank, or build from scratch
    feature_bank_dir = cfg["evaluation"].get("output_dir", "results/")
    bank = FeatureBank(
        extractor=extractor,
        categories=categories,
        reduce_dim=reduce_dim,
        device=device,
    )

    bank_path = os.path.join(feature_bank_dir, "feature_bank.pkl")
    if os.path.exists(bank_path):
        print(f"Loading pre-built feature bank from {bank_path}")
        bank.load(feature_bank_dir)
    else:
        print("Building feature bank from scratch (this will take time)...")
        bank.build(
            root=data_root,
            image_size=img_size,
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
        )
        bank.save(feature_bank_dir)

    # PaDiM detector
    top_k_ratio = cfg["anomaly"].get("top_k", 0.01)
    detector = PaDiMDetector(
        extractor=extractor,
        means=bank.means,
        cov_invs=bank.cov_invs,
        proj_matrices=bank.proj_matrices,
        top_k_ratio=top_k_ratio,
    )

    # Evaluator
    evaluator = AnomalyEvaluator(
        extractor=extractor,
        bank=bank,
        detector=detector,
        cfg=cfg,
    )

    # Run evaluation
    results = evaluator.evaluate_all()

    # Save results
    output_dir = cfg["evaluation"].get("output_dir", "results/")
    evaluator.save_results(results, output_dir)


if __name__ == "__main__":
    main()
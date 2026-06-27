"""
Anomaly detection evaluation pipeline.

Runs per-category anomaly detection evaluation using pretrained I-JEPA features
and PaDiM-style distribution modeling. Outputs image-level and pixel-level AUROC.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..features.extract import FeatureExtractor
from ..features.bank import FeatureBank
from ..anomaly.padim import PaDiMDetector
from ..data.dataset import AnomalyDataset, _discover_categories
from ..data.augmentations import IJEPAAugmentations


class AnomalyEvaluator:
    """
    End-to-end anomaly detection evaluation.

    Args:
        extractor: FeatureExtractor instance.
        bank: FeatureBank instance (pre-built or to be built).
        detector: PaDiMDetector instance.
        cfg: Evaluation configuration dict.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        bank: FeatureBank,
        detector: PaDiMDetector,
        cfg: dict,
    ):
        self.extractor = extractor
        self.bank = bank
        self.detector = detector
        self.cfg = cfg

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate anomaly detection for all categories.

        Returns:
            results: {category: {"image_auroc": float, "pixel_auroc": float}, ...}
        """
        data_root = os.path.expanduser(self.cfg["data"]["root"])
        img_size = self.cfg["data"]["image_size"]
        batch_size = self.cfg["data"]["batch_size"]

        categories = self.cfg["evaluation"].get("categories", [])
        if not categories:
            categories = _discover_categories(data_root)

        augmentations = IJEPAAugmentations(image_size=img_size)

        all_results = {}

        for category in tqdm(categories, desc="Evaluating all categories"):
            # Build feature bank if not already built
            if category not in self.bank.means:
                self._build_bank_for_category(
                    category, data_root, img_size, batch_size,
                    self.cfg["data"]["num_workers"], augmentations,
                )

            # Load test data
            test_dataset = AnomalyDataset(
                root=data_root,
                category=category,
                split="test",
                transform=augmentations.get_eval(),
            )

            if len(test_dataset) == 0:
                print(f"  [WARN] No test data for {category}, skipping")
                continue

            # Evaluate
            metrics = self.detector.evaluate_category(
                test_images=[test_dataset[i]["image"] for i in range(len(test_dataset))],
                test_labels=[test_dataset[i]["label"] for i in range(len(test_dataset))],
                test_masks=[test_dataset[i]["mask"] for i in range(len(test_dataset))],
                category=category,
            )

            all_results[category] = metrics
            print(
                f"  [{category}] Image AUROC: {metrics['image_auroc']:.4f}, "
                f"Pixel AUROC: {metrics['pixel_auroc']:.4f} "
                f"({metrics['num_normal']}N + {metrics['num_anomaly']}A)"
            )

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _build_bank_for_category(
        self, category, data_root, img_size, batch_size, num_workers, augmentations,
    ):
        """Build feature bank for a single category on-demand."""
        print(f"  Building feature bank for {category}...")

        train_dataset = AnomalyDataset(
            root=data_root,
            category=category,
            split="train",
            transform=augmentations.get_eval(),
        )

        if len(train_dataset) == 0:
            raise RuntimeError(f"No training images for {category}")

        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        all_features = []
        for batch in loader:
            images = batch["image"]
            batch_features = self.extractor.extract_batch(images)
            all_features.extend(batch_features)

        features = np.stack(all_features, axis=0)  # (S, P, D)
        self.bank.banks[category] = features
        self.bank._estimate_distributions(category)

    def _print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print summary table of results."""
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION SUMMARY")
        print("=" * 60)
        print(f"{'Category':<30} {'Image AUROC':>12} {'Pixel AUROC':>12}")
        print("-" * 56)

        img_aurocs = []
        pixel_aurocs = []

        for cat, metrics in sorted(results.items()):
            ia = metrics["image_auroc"]
            pa = metrics["pixel_auroc"]
            img_aurocs.append(ia)
            pixel_aurocs.append(pa)
            print(f"{cat:<30} {ia:>12.4f} {pa:>12.4f}")

        print("-" * 56)
        mean_ia = np.mean(img_aurocs) if img_aurocs else 0.0
        mean_pa = np.mean(pixel_aurocs) if pixel_aurocs else 0.0
        print(f"{'MEAN':<30} {mean_ia:>12.4f} {mean_pa:>12.4f}")
        print("=" * 60)

    def save_results(self, results: Dict, output_dir: str):
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "anomaly_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path / 'anomaly_results.json'}")
"""
Feature bank — collect and store normal sample features per category.

For each component category, we collect patch features from all normal
training images. These are used to estimate a multivariate Gaussian
distribution (per patch position) for anomaly detection.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

from .extract import FeatureExtractor
from ..data.dataset import AnomalyDataset
from ..data.augmentations import IJEPAAugmentations
from torch.utils.data import DataLoader


class FeatureBank:
    """
    Collects and manages normal-sample patch features per category.

    Args:
        extractor: FeatureExtractor instance.
        categories: List of category names.
        reduce_dim: Dimensionality reduction target (random projection, PaDiM-style).
        device: torch device.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        categories: List[str],
        reduce_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.extractor = extractor
        self.categories = categories
        self.reduce_dim = reduce_dim
        self.device = device or extractor.device

        # per-category feature banks
        self.banks: Dict[str, np.ndarray] = {}  # category → (num_patches, num_samples, feature_dim)
        self.means: Dict[str, np.ndarray] = {}  # category → (num_patches, feature_dim)
        self.cov_invs: Dict[str, np.ndarray] = {}  # category → (num_patches, feature_dim, feature_dim)
        self.proj_matrices: Dict[str, np.ndarray] = {}  # category → (raw_dim, reduce_dim)

    def build(
        self,
        root: str,
        image_size: int = 224,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        """
        Build feature banks for all categories.

        Args:
            root: Dataset root path.
            image_size: Input image size.
            batch_size: Batch size for feature extraction.
            num_workers: DataLoader workers.
        """
        augmentations = IJEPAAugmentations(image_size=image_size)

        for category in tqdm(self.categories, desc="Building feature banks"):
            print(f"\n  Processing [{category}]...")

            # Load normal training images for this category
            dataset = AnomalyDataset(
                root=root,
                category=category,
                split="train",
                transform=augmentations.get_eval(),
            )

            if len(dataset) == 0:
                print(f"  [WARN] No training images for {category}, skipping")
                continue

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory="cuda" in str(self.device),
            )

            # Collect patch features from all normal images
            all_features = []  # list of (num_patches, D)

            for batch in tqdm(loader, desc=f"Extracting {category}", leave=False):
                images = batch["image"]
                batch_features = self.extractor.extract_batch(images)
                all_features.extend(batch_features)

            # Stack: (num_samples, num_patches, D) → rearrange to (num_patches, num_samples, D)
            features = np.stack(all_features, axis=0)  # (S, P, D)
            self.banks[category] = features.copy()

            # Estimate Gaussian: mean + covariance per patch position
            self._estimate_distributions(category)

            print(f"  [{category}] {features.shape[0]} images, "
                  f"{features.shape[1]} patches, dim={features.shape[2]}")

    def _estimate_distributions(self, category: str):
        """
        Estimate per-patch Gaussian distributions.

        For each patch position, fit a multivariate Gaussian to the
        features across all normal training samples.

        Uses Ledoit-Wolf shrinkage for robust covariance estimation when
        num_samples < feature_dim (which is common in anomaly detection).
        """
        from sklearn.covariance import LedoitWolf

        features = self.banks[category]  # (S, P, D)
        S, P, D = features.shape

        # Optional dimensionality reduction
        if self.reduce_dim and self.reduce_dim < D:
            # Random projection + PCA-style: use a random orthonormal projection
            rng = np.random.RandomState(42)
            proj = rng.randn(D, self.reduce_dim)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            self.proj_matrices[category] = proj

            # Reshape for projection
            features_2d = features.transpose(1, 0, 2).reshape(P * S, D)  # (P*S, D)
            features_proj = features_2d @ proj  # (P*S, R)
            features = features_proj.reshape(P, S, self.reduce_dim).transpose(1, 0, 2)  # (S, P, R)
            D = self.reduce_dim

        # Per-patch mean and covariance
        means = np.zeros((P, D))
        cov_invs = np.zeros((P, D, D))

        for p in range(P):
            patch_features = features[:, p, :]  # (S, D)

            # Mean
            means[p] = patch_features.mean(axis=0)

            # Covariance with shrinkage
            if S > 1:
                try:
                    lw = LedoitWolf().fit(patch_features)
                    cov = lw.covariance_
                    # Regularized inverse
                    reg = 0.01 * np.trace(cov) / D
                    cov_reg = cov + reg * np.eye(D)
                    cov_invs[p] = np.linalg.inv(cov_reg)
                except Exception:
                    # Fallback: diagonal covariance
                    var = patch_features.var(axis=0) + 1e-6
                    cov_invs[p] = np.diag(1.0 / var)
            else:
                var = np.ones(D) * 1e-6
                cov_invs[p] = np.diag(1.0 / var)

        self.means[category] = means
        self.cov_invs[category] = cov_invs

    def save(self, path: str):
        """Save all banks to a directory."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "banks": {cat: arr for cat, arr in self.banks.items()},
            "means": {cat: arr for cat, arr in self.means.items()},
            "cov_invs": {cat: arr for cat, arr in self.cov_invs.items()},
            "proj_matrices": {cat: arr for cat, arr in self.proj_matrices.items()},
            "categories": self.categories,
            "reduce_dim": self.reduce_dim,
        }

        for cat in self.categories:
            cat_file = save_dir / f"{cat}.pkl"
            cat_data = {
                "mean": self.means.get(cat),
                "cov_inv": self.cov_invs.get(cat),
                "proj_matrix": self.proj_matrices.get(cat),
                "num_samples": len(self.banks.get(cat, [])),
            }
            with open(cat_file, "wb") as f:
                pickle.dump(cat_data, f)

        # Also save full combined file
        with open(save_dir / "feature_bank.pkl", "wb") as f:
            pickle.dump(data, f)

        print(f"Feature banks saved to {save_dir}")

    def load(self, path: str):
        """Load all banks from a directory."""
        load_path = Path(path)

        # Try combined file first
        combined = load_path / "feature_bank.pkl"
        if combined.exists():
            with open(combined, "rb") as f:
                data = pickle.load(f)
            self.banks = data["banks"]
            self.means = data["means"]
            self.cov_invs = data["cov_invs"]
            self.proj_matrices = data.get("proj_matrices", {})
            self.categories = data.get("categories", list(self.banks.keys()))
            print(f"Loaded feature bank from {combined}")
            return

        # Load per-category files
        for cat in self.categories:
            cat_file = load_path / f"{cat}.pkl"
            if not cat_file.exists():
                print(f"  [WARN] No bank file for {cat}")
                continue
            with open(cat_file, "rb") as f:
                cat_data = pickle.load(f)
            self.means[cat] = cat_data["mean"]
            self.cov_invs[cat] = cat_data["cov_inv"]
            if "proj_matrix" in cat_data:
                self.proj_matrices[cat] = cat_data["proj_matrix"]

        print(f"Loaded feature banks from {load_path} ({len(self.means)} categories)")

    def get_distribution(self, category: str):
        """Get (means, cov_invs, proj_matrix) for a category."""
        return (
            self.means.get(category),
            self.cov_invs.get(category),
            self.proj_matrices.get(category),
        )
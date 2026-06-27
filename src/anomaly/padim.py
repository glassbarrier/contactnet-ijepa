"""
PaDiM-style anomaly detector using I-JEPA features.

For each patch position, the normal features follow a multivariate Gaussian
distribution. Anomaly score = Mahalanobis distance from the learned distribution.

Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework
for Anomaly Detection and Localization", ICPR 2021.
"""

from typing import Dict, Tuple, Optional

import torch
import numpy as np
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm

from ..features.extract import FeatureExtractor


class PaDiMDetector:
    """
    Patch Distribution Modeling anomaly detector.

    Args:
        extractor: FeatureExtractor instance.
        means: Per-category per-patch means from feature bank.
        cov_invs: Per-category per-patch inverse covariance matrices.
        proj_matrices: Per-category projection matrices (optional).
        top_k_ratio: Ratio of top anomaly scores used for image-level score.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        means: Dict[str, np.ndarray],
        cov_invs: Dict[str, np.ndarray],
        proj_matrices: Optional[Dict[str, np.ndarray]] = None,
        top_k_ratio: float = 0.01,
    ):
        self.extractor = extractor
        self.means = means
        self.cov_invs = cov_invs
        self.proj_matrices = proj_matrices or {}
        self.top_k_ratio = top_k_ratio

    def score_image(
        self,
        image: torch.Tensor,
        category: str,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute anomaly score map and image-level score for a single image.

        Args:
            image: (1, 3, H, W) normalized tensor.
            category: Component category name.

        Returns:
            anomaly_map: (grid_h, grid_w) anomaly scores per patch.
            image_score: Scalar image-level anomaly score.
        """
        # Extract features
        features = self.extractor.extract_features(image)  # (num_patches, D)

        # Apply projection if available
        if category in self.proj_matrices:
            features = features @ self.proj_matrices[category]

        # Get distribution parameters
        means = self.means.get(category)  # (num_patches, D)
        cov_invs = self.cov_invs.get(category)  # (num_patches, D, D)

        if means is None or cov_invs is None:
            raise ValueError(f"No distribution parameters for category '{category}'")

        num_patches, D = features.shape
        grid_h = grid_w = int(np.sqrt(num_patches))

        # Compute Mahalanobis distance per patch
        anomaly_map = np.zeros(num_patches)

        for p in range(num_patches):
            diff = features[p] - means[p]
            mahal = np.sqrt(diff @ cov_invs[p] @ diff)
            anomaly_map[p] = mahal

        # Reshape to spatial grid
        anomaly_map = anomaly_map.reshape(grid_h, grid_w)

        # Image-level score: mean of top-K anomaly scores
        top_k = max(1, int(self.top_k_ratio * num_patches))
        top_scores = np.sort(anomaly_map.flatten())[-top_k:]
        image_score = top_scores.mean()

        return anomaly_map, image_score

    def score_batch(
        self,
        images: torch.Tensor,
        category: str,
    ) -> Tuple[list, np.ndarray]:
        """
        Score a batch of images.

        Returns:
            anomaly_maps: list of (H, W) arrays.
            image_scores: (B,) array.
        """
        anomaly_maps = []
        image_scores = []

        for i in range(images.shape[0]):
            amap, iscore = self.score_image(images[i:i+1], category)
            anomaly_maps.append(amap)
            image_scores.append(iscore)

        return anomaly_maps, np.array(image_scores)

    def evaluate_category(
        self,
        test_images: list,
        test_labels: list,
        test_masks: list,
        category: str,
    ) -> Dict[str, float]:
        """
        Evaluate anomaly detection on a single category.

        Args:
            test_images: list of (1, 3, H, W) tensors.
            test_labels: list of int (0=normal, 1=anomaly).
            test_masks: list of (1, H, W) tensors (ground truth pixel masks).
            category: Category name.

        Returns:
            Dict with image_auroc, pixel_auroc.
        """
        from ..utils.metrics import compute_image_auroc, compute_pixel_auroc

        all_anomaly_maps = []
        all_image_scores = []
        all_labels = []
        all_gt_masks = []

        for img, label, mask in tqdm(
            zip(test_images, test_labels, test_masks),
            desc=f"Evaluating {category}",
            total=len(test_images),
        ):
            amap, iscore = self.score_image(img, category)
            all_anomaly_maps.append(amap)
            all_image_scores.append(iscore)
            all_labels.append(label)
            all_gt_masks.append(mask)

        # Compute metrics
        image_auroc = compute_image_auroc(
            np.array(all_image_scores), np.array(all_labels)
        )

        # Pixel AUROC: need to upsample anomaly maps to image resolution
        pixel_auroc = self._compute_pixel_auroc(
            all_anomaly_maps, all_gt_masks
        )

        return {
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "num_normal": sum(1 for l in all_labels if l == 0),
            "num_anomaly": sum(1 for l in all_labels if l == 1),
        }

    def _compute_pixel_auroc(
        self,
        anomaly_maps: list,
        gt_masks: list,
        img_size: int = 224,
    ) -> float:
        """
        Compute pixel-level AUROC by upsampling patch-grid anomaly maps
        to the image resolution.
        """
        from ..utils.metrics import compute_pixel_auroc
        import cv2

        upsampled_maps = []
        upsampled_masks = []

        for amap, mask in zip(anomaly_maps, gt_masks):
            # Upsample anomaly map from grid size to image size
            grid_h, grid_w = amap.shape
            amap_upsampled = cv2.resize(
                amap, (img_size, img_size),
                interpolation=cv2.INTER_LINEAR,
            )

            # Ground truth mask (already at img_size or similar)
            mask_np = mask.squeeze().cpu().numpy()  # (H, W)
            if mask_np.shape != (img_size, img_size):
                mask_np = cv2.resize(
                    mask_np, (img_size, img_size),
                    interpolation=cv2.INTER_NEAREST,
                )

            upsampled_maps.append(amap_upsampled)
            upsampled_masks.append(mask_np)

        return compute_pixel_auroc(upsampled_maps, upsampled_masks)
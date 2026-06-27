"""
Evaluation metrics for anomaly detection.

Computes image-level and pixel-level AUROC scores.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, auc
from typing import Tuple


def compute_image_auroc(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute image-level AUROC.

    Args:
        anomaly_scores: (N,) array of anomaly scores (higher = more anomalous).
        labels: (N,) array of binary labels (0 = normal, 1 = anomaly).

    Returns:
        AUROC score.
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # Undefined when only one class present

    return roc_auc_score(labels, anomaly_scores)


def compute_pixel_auroc(
    anomaly_maps: list,
    gt_masks: list,
) -> float:
    """
    Compute pixel-level AUROC.

    Args:
        anomaly_maps: list of (H, W) anomaly score maps.
        gt_masks: list of (H, W) binary ground truth masks (0 = normal, 1 = anomaly).

    Returns:
        Pixel-level AUROC score.
    """
    all_scores = np.concatenate([am.flatten() for am in anomaly_maps])
    all_labels = np.concatenate([gm.flatten() for gm in gt_masks])

    if len(np.unique(all_labels)) < 2:
        return 0.5

    return roc_auc_score(all_labels, all_scores)


def compute_optimal_threshold(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Find the optimal threshold maximizing F1 for image-level detection.

    Returns threshold value.
    """
    if len(np.unique(labels)) < 2:
        return float(np.median(anomaly_scores))

    best_threshold = 0.0
    best_f1 = 0.0

    # Search thresholds between min and max
    thresholds = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 200)

    for thr in thresholds:
        preds = (anomaly_scores >= thr).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    return best_threshold
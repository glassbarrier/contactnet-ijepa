"""
MVTec-AD style dataset loader for contact network components.

Dataset structure:
    <root>/
        <category>/
            train/good/         # Normal images for training
            test/good/          # Normal images for testing
            test/<defect_type>/ # Anomalous images for testing
            ground_truth/<defect_type>/  # Pixel-level anomaly masks

Special case: ear_croped has both `train/good` and `train/二分类good`.
We only use `train/good` (skip 二分类good).
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


def _discover_categories(root: str) -> List[str]:
    """
    Auto-discover all component categories in the dataset root.

    A valid category must have `train/good/` subdirectory.
    Returns sorted list of category names.
    """
    root = Path(os.path.expanduser(root))
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    categories = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            train_good = entry / "train" / "good"
            if train_good.is_dir():
                categories.append(entry.name)

    return categories


def _image_extensions() -> Tuple[str, ...]:
    return (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def _list_images(dir_path: Path) -> List[str]:
    """List all image files in a directory."""
    images = []
    for ext in _image_extensions():
        images.extend(glob.glob(str(dir_path / f"*{ext}")))
        images.extend(glob.glob(str(dir_path / f"*{ext.upper()}")))
    return sorted(images)


def _get_defect_types(category_path: Path) -> List[str]:
    """Discover defect types for a category from test/ directory."""
    test_path = category_path / "test"
    if not test_path.exists():
        return []

    defects = []
    for entry in sorted(test_path.iterdir()):
        if entry.is_dir() and entry.name != "good":
            defects.append(entry.name)
    return defects


class PretrainDataset(Dataset):
    """
    Dataset for I-JEPA pretraining.

    Loads ALL `train/good` images from ALL categories (or a subset).
    No labels are used during pretraining — just images + masks.

    Args:
        root: Path to MVTec-AD style dataset root.
        categories: List of categories to include. Empty = auto-discover all.
        transform: torchvision transform to apply to each image.
    """

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        transform=None,
    ):
        self.root = Path(os.path.expanduser(root))
        self.transform = transform

        if categories is None or len(categories) == 0:
            categories = _discover_categories(str(self.root))

        self.categories = categories
        self.image_paths: List[str] = []

        for cat in self.categories:
            cat_train_good = self.root / cat / "train" / "good"
            if not cat_train_good.is_dir():
                print(f"  [WARN] Skipping {cat}: no train/good/ found")
                continue

            cat_images = _list_images(cat_train_good)
            # Skip files that contain '二分类' in path (ear_croped special case)
            cat_images = [
                img for img in cat_images
                if "二分类" not in img and "二分类" not in str(Path(img).parent)
            ]
            self.image_paths.extend(cat_images)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No training images found in {self.root}. "
                f"Checked categories: {self.categories}"
            )

        print(f"PretrainDataset: {len(self.image_paths)} images from {len(self.categories)} categories")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


class AnomalyDataset(Dataset):
    """
    Dataset for anomaly detection evaluation on a SINGLE category.

    Loads:
      - train/good: normal reference images (for feature bank construction)
      - test/good: normal test images
      - test/<defect_type>: anomalous test images
      - ground_truth/<defect_type>: pixel-level masks

    Returns (image, label, mask, category_name):
        label: 0 = normal (good), 1 = anomaly
        mask: pixel-level ground truth (0 = normal, 1 = anomaly), or None if normal
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "test",  # "train" or "test"
        transform=None,
    ):
        self.root = Path(os.path.expanduser(root))
        self.category = category
        self.split = split
        self.transform = transform

        self.cat_path = self.root / category
        if not self.cat_path.exists():
            raise FileNotFoundError(f"Category path not found: {self.cat_path}")

        # Collect image paths with labels and mask paths
        self.samples: List[Dict] = []

        if split == "train":
            # Only good/normal images for training
            train_good = self.cat_path / "train" / "good"
            if train_good.is_dir():
                for img_path in _list_images(train_good):
                    if "二分类" in str(img_path) or "二分类" in str(Path(img_path).parent):
                        continue
                    self.samples.append({
                        "image": img_path,
                        "label": 0,  # normal
                        "mask": None,
                    })
        else:
            # test split: good + all defect types
            # Normal test images
            test_good = self.cat_path / "test" / "good"
            if test_good.is_dir():
                for img_path in _list_images(test_good):
                    self.samples.append({
                        "image": img_path,
                        "label": 0,  # normal
                        "mask": None,
                    })

            # Anomalous test images
            defect_types = _get_defect_types(self.cat_path)
            for defect in defect_types:
                test_defect = self.cat_path / "test" / defect
                gt_defect = self.cat_path / "ground_truth" / defect

                if not test_defect.is_dir():
                    continue

                defect_images = _list_images(test_defect)
                for img_path in defect_images:
                    # Look for corresponding ground truth mask
                    mask_path = self._find_mask(img_path, gt_defect)
                    self.samples.append({
                        "image": img_path,
                        "label": 1,  # anomaly
                        "mask": mask_path,
                    })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found for category={category}, split={split}"
            )

        # Print summary
        n_normal = sum(1 for s in self.samples if s["label"] == 0)
        n_anomaly = sum(1 for s in self.samples if s["label"] == 1)
        print(f"AnomalyDataset [{category}/{split}]: {n_normal} normal, {n_anomaly} anomaly")

    def _find_mask(self, img_path: str, gt_dir: Path) -> Optional[str]:
        """Find the corresponding ground truth mask for an anomaly image."""
        if not gt_dir.is_dir():
            return None

        img_stem = Path(img_path).stem

        # Try direct match first
        for ext in _image_extensions():
            candidate = gt_dir / f"{img_stem}{ext}"
            if candidate.exists():
                return str(candidate)
            candidate = gt_dir / f"{img_stem}{ext.upper()}"
            if candidate.exists():
                return str(candidate)

        # Try suffix patterns (_mask, _gt)
        for suffix in ["_mask", "_gt"]:
            for ext in _image_extensions():
                candidate = gt_dir / f"{img_stem}{suffix}{ext}"
                if candidate.exists():
                    return str(candidate)

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        image = Image.open(sample["image"]).convert("RGB")
        image_size = image.size  # (W, H) before transform

        if self.transform is not None:
            image = self.transform(image)

        result = {
            "image": image,
            "label": sample["label"],
            "category": self.category,
            "image_path": sample["image"],
        }

        # Load mask if present
        if sample["mask"] is not None:
            mask = Image.open(sample["mask"]).convert("L")
            if self.transform is not None:
                # Apply same geometric transform to mask (resize + center crop only, no color transforms)
                from torchvision.transforms import functional as F
                # Resize to 1.14x then center crop, matching eval transform
                h, w = mask.height, mask.width
                target_size = int(224 * 1.14)
                mask = F.resize(
                    mask, target_size,
                    interpolation=F.InterpolationMode.NEAREST,
                )
                mask = F.center_crop(mask, 224)
                mask = F.to_tensor(mask)  # (1, H, W) in [0, 1]
                mask = (mask > 0.5).float()
            result["mask"] = mask
        else:
            # Normal image: mask is all zeros
            _, H, W = image.shape
            result["mask"] = torch.zeros(1, H, W)

        return result
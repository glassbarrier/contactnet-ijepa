"""
Data augmentation pipeline for I-JEPA pretraining.

I-JEPA uses relatively mild augmentations compared to contrastive methods.
Key augmentations: random resized crop, horizontal flip, color jitter.

Unlike MAE/MoCo/SimCLR, I-JEPA does NOT need heavy color augmentation because
the task is representation prediction, not invariance learning.
"""

import torchvision.transforms as T
from typing import Tuple


class IJEPAAugmentations:
    """
    Augmentation pipeline for I-JEPA training.

    The same augmentations are applied to the full image before mask sampling,
    so context and target blocks come from the same augmented view.

    Args:
        image_size: Target crop size (square).
        crop_scale: (min, max) scale range for RandomResizedCrop.
        hflip_prob: Probability of horizontal flip.
        color_jitter_strength: Max brightness/contrast/saturation jitter.
    """

    def __init__(
        self,
        image_size: int = 224,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        hflip_prob: float = 0.5,
        color_jitter_strength: float = 0.4,
    ):
        self.train_transform = T.Compose([
            T.RandomResizedCrop(
                image_size,
                scale=crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=hflip_prob),
            T.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=0.1,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # Test/eval transform: no augmentation, just resize + normalize
        self.eval_transform = T.Compose([
            T.Resize(int(image_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def get_train(self) -> T.Compose:
        return self.train_transform

    def get_eval(self) -> T.Compose:
        return self.eval_transform


# Default ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
"""
Feature extraction from pretrained I-JEPA context encoder.

Extracts multi-scale patch features for downstream anomaly detection.
Similar to PaDiM: features from multiple encoder layers are concatenated,
giving richer representations for anomaly localization.
"""

from typing import List

import torch
import torch.nn.functional as F
import numpy as np

from ..models.vision_transformer import VisionTransformer


class FeatureExtractor:
    """
    Extract multi-scale patch features from pretrained ViT.

    Args:
        encoder: Pretrained context encoder (VisionTransformer).
        layer_indices: Which transformer block outputs to use (1-indexed).
        patch_size: ViT patch size.
        img_size: Input image size.
        device: torch device.
    """

    def __init__(
        self,
        encoder: VisionTransformer,
        layer_indices: List[int] = None,
        patch_size: int = 16,
        img_size: int = 224,
        device: torch.device = None,
    ):
        if layer_indices is None:
            layer_indices = [4, 6, 8, 10, 12]

        self.encoder = encoder
        # Convert 1-indexed layer numbers to 0-indexed
        self.layer_indices = [i - 1 for i in layer_indices]
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder.to(self.device)
        self.encoder.eval()

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> np.ndarray:
        """
        Extract multi-scale patch features from a single image.

        Args:
            image: (1, 3, H, W) normalized tensor.

        Returns:
            features: (num_patches, total_embed_dim) numpy array.
        """
        image = image.to(self.device)
        B = image.shape[0]

        # Patch embed
        patches = self.encoder.patch_embed(image)  # (B, N, D)
        patches = patches + self.encoder.pos_embed[:, :self.encoder.num_patches, :]

        # Add CLS token
        cls_token = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, patches], dim=1)  # (B, 1+N, D)

        # Collect features from specified layers
        features_list = []

        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.layer_indices:
                patch_feat = x[:, 1:, :]  # (B, N, D) — skip CLS
                # L2 normalize per layer (PaDiM-style)
                patch_feat = F.normalize(patch_feat, p=2, dim=-1)
                features_list.append(patch_feat.squeeze(0))  # (N, D)

        # Concatenate across layers → (N, total_dim)
        features = torch.cat(features_list, dim=-1)  # (N, total_D)
        return features.cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> list:
        """
        Extract features for a batch of images.

        Args:
            images: (B, 3, H, W)

        Returns:
            list of (num_patches, total_embed_dim) arrays.
        """
        return [self.extract_features(images[i:i+1]) for i in range(images.shape[0])]
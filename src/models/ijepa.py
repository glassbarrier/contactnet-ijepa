"""
I-JEPA Model — full assembly.

I-JEPA learns semantic image representations by predicting the latent features
of masked target image blocks from visible context blocks.

Components:
    - Context Encoder: ViT that processes visible context blocks
    - Target Encoder: ViT (EMA of context encoder) that processes target blocks
    - Predictor: Narrower transformer that predicts target representations

The target encoder is updated via EMA (Exponential Moving Average) of the
context encoder weights — never via gradient descent.
"""

import copy
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import VisionTransformer
from .predictor import IJEPAPredictor


class IJPEModel(nn.Module):
    """
    Full I-JEPA model combining context encoder, target encoder, and predictor.

    Args:
        img_size: Input image size.
        patch_size: ViT patch size.
        context_embed_dim: Context/target encoder embedding dimension.
        context_depth: Number of transformer blocks in encoders.
        context_heads: Number of attention heads in encoders.
        predictor_embed_dim: Predictor internal embedding dimension.
        predictor_depth: Number of transformer blocks in predictor.
        predictor_heads: Number of attention heads in predictor.
        mlp_ratio: MLP hidden ratio for encoders and predictor.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        context_embed_dim: int = 384,
        context_depth: int = 12,
        context_heads: int = 6,
        predictor_embed_dim: int = 192,
        predictor_depth: int = 6,
        predictor_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Context encoder (trained by gradient descent)
        self.context_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=context_embed_dim,
            depth=context_depth,
            num_heads=context_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Target encoder (EMA of context encoder — no gradients)
        self.target_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=context_embed_dim,
            depth=context_depth,
            num_heads=context_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        # Initialize target encoder as a copy of context encoder
        self.target_encoder.load_state_dict(
            copy.deepcopy(self.context_encoder.state_dict())
        )
        # Stop gradients for target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = IJEPAPredictor(
            context_embed_dim=context_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            target_embed_dim=context_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches=self.num_patches,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward_context(
        self,
        images: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extract context features from visible patches.

        Args:
            images: (B, 3, H, W)
            context_mask: (B, num_patches) boolean — True = visible

        Returns:
            List of (N_vis, D) context features per batch item.
        """
        return self.context_encoder.forward_masked(images, context_mask)

    def forward_target(
        self,
        images: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extract target features from target patches (no grad).

        Args:
            images: (B, 3, H, W)
            target_mask: (B, num_patches) boolean — True = target

        Returns:
            List of (N_tgt, D) target features per batch item.
        """
        with torch.no_grad():
            return self.target_encoder.forward_masked(images, target_mask)

    def forward_predictor(
        self,
        context_features: List[torch.Tensor],
        context_patch_indices: List[torch.Tensor],
        target_patch_indices: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Predict target features from context features.

        Args:
            context_features: List of (N_ctx, D) per batch item.
            context_patch_indices: List of (N_ctx,) long — which patches.
            target_patch_indices: List of (N_tgt,) long — which patches to predict.

        Returns:
            List of (N_tgt, D) predicted features per batch item.
        """
        return self.predictor(
            context_features, context_patch_indices, target_patch_indices
        )

    def compute_loss(
        self,
        pred_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute L2 loss between predicted and target features.

        Args:
            pred_features: List of (N_tgt, D) per batch item.
            target_features: List of (N_tgt, D) per batch item.

        Returns:
            Scalar loss (mean L2 across all target patches).
        """
        total_loss = torch.tensor(0.0, device=pred_features[0].device)
        total_patches = 0

        for pred, target in zip(pred_features, target_features):
            if len(pred) == 0 or len(target) == 0:
                continue
            # Sum of squared errors (differentiable); normalize at end
            loss = F.mse_loss(pred, target, reduction='sum')
            total_loss = total_loss + loss
            total_patches += len(pred)

        if total_patches == 0:
            return torch.tensor(0.0, device=pred_features[0].device)

        return total_loss / total_patches

    def forward(
        self,
        images: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Full I-JEPA forward pass.

        Args:
            images: (B, 3, H, W)
            context_mask: (B, num_patches) boolean — context (visible) patches
            target_mask: (B, num_patches) boolean — target patches for ONE block

        Returns:
            context_features: List of (N_ctx, D) per batch item
            target_features: List of (N_tgt, D) per batch item
            loss: Scalar L2 loss
        """
        B = images.shape[0]

        # Get patch indices from masks
        all_context_indices = []
        all_target_indices = []
        for b in range(B):
            ctx_idx = context_mask[b].nonzero(as_tuple=True)[0]
            tgt_idx = target_mask[b].nonzero(as_tuple=True)[0]
            all_context_indices.append(ctx_idx)
            all_target_indices.append(tgt_idx)

        # Context encoder forward
        context_features = self.forward_context(images, context_mask)

        # Target encoder forward
        target_features = self.forward_target(images, target_mask)

        # Predictor forward
        pred_features = self.forward_predictor(
            context_features, all_context_indices, all_target_indices
        )

        # Loss
        loss = self.compute_loss(pred_features, target_features)

        return context_features, target_features, loss

    def ema_update(self, momentum: float):
        """
        Update target encoder via EMA of context encoder.

        target_param = momentum * target_param + (1 - momentum) * context_param
        """
        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_param.data.mul_(momentum).add_(ctx_param.data, alpha=1.0 - momentum)

    def get_context_encoder(self) -> VisionTransformer:
        """Return the context encoder for downstream feature extraction."""
        return self.context_encoder
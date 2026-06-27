"""
I-JEPA Predictor network.

The predictor takes context encoder outputs + target position information,
and predicts the target encoder's representation for each target block.

It is a narrower/shallower Transformer that maps:
    context_features → predicted_target_features

Architecture:
    - Learnable mask tokens for target positions
    - Shallow Transformer blocks
    - Input projection (context_dim → predictor_dim)
    - Output projection (predictor_dim → target_dim)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import TransformerBlock


class IJEPAPredictor(nn.Module):
    """
    Predictor network for I-JEPA.

    Takes context encoder features and predicts target encoder features.

    Args:
        context_embed_dim: Embedding dimension of the context encoder output.
        predictor_embed_dim: Internal embedding dimension (smaller for efficiency).
        target_embed_dim: Embedding dimension to predict (same as context_embed_dim usually).
        depth: Number of transformer blocks in predictor.
        num_heads: Attention heads.
        num_patches: Total number of patches in the image grid (for mask token setup).
        mlp_ratio: MLP hidden ratio.
    """

    def __init__(
        self,
        context_embed_dim: int = 384,
        predictor_embed_dim: int = 192,
        target_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 3,
        num_patches: int = 196,  # 14x14 for 224/16
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.predictor_embed_dim = predictor_embed_dim

        # Learnable mask token — shared across all target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Learnable positional embeddings for target positions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim)
        )

        # Input projection: context features → predictor dim
        self.input_proj = nn.Linear(context_embed_dim, predictor_embed_dim, bias=True)

        # Transformer blocks (narrower)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)

        # Output projection: predictor dim → target embed dim
        self.output_proj = nn.Linear(predictor_embed_dim, target_embed_dim, bias=True)

        # Initialize
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        context_features: List[torch.Tensor],
        context_patch_indices: List[torch.Tensor],
        target_patch_indices: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Predict target features from context features.

        Args:
            context_features: List of (N_ctx, D_ctx) tensors — one per batch item.
            context_patch_indices: List of (N_ctx,) long tensors — which patches
                                   correspond to each context feature.
            target_patch_indices: List of (N_tgt,) long tensors — which patches
                                  to predict for each batch item (ONE target block).

        Returns:
            List of (N_tgt, D_target) predicted features — one per batch item.
        """
        batch_size = len(context_features)
        predictions = []

        for b in range(batch_size):
            ctx_feat = context_features[b]  # (N_ctx, D_ctx)
            ctx_idx = context_patch_indices[b]  # (N_ctx,)
            tgt_idx = target_patch_indices[b]  # (N_tgt,)

            if len(ctx_feat) == 0 or len(tgt_idx) == 0:
                predictions.append(torch.empty(
                    0, self.output_proj.out_features,
                    device=ctx_feat.device,
                ))
                continue

            # Project context features to predictor dim
            ctx_feat_p = self.input_proj(ctx_feat)  # (N_ctx, D_pred)

            # Create mask tokens for target positions
            num_tgt = len(tgt_idx)
            mask_tokens = self.mask_token.expand(num_tgt, -1)  # (N_tgt, D_pred)

            # Add positional embeddings
            ctx_feat_p = ctx_feat_p + self.pos_embed[0, ctx_idx.long()]  # (N_ctx, D_pred)
            mask_tokens = mask_tokens + self.pos_embed[0, tgt_idx.long()]  # (N_tgt, D_pred)

            # Concatenate: [context features, mask tokens]
            tokens = torch.cat([ctx_feat_p, mask_tokens], dim=0)  # (N_ctx+N_tgt, D_pred)
            tokens = tokens.unsqueeze(0)  # (1, N_all, D_pred)

            # Run through predictor transformer
            for block in self.blocks:
                tokens = block(tokens)

            tokens = self.norm(tokens)

            # Extract only the target positions (last N_tgt tokens)
            tgt_predictions = tokens[0, -num_tgt:]  # (N_tgt, D_pred)

            # Project to target embedding dimension
            tgt_predictions = self.output_proj(tgt_predictions)  # (N_tgt, D_target)

            predictions.append(tgt_predictions)

        return predictions
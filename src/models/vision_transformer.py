"""
Vision Transformer (ViT) backbone for I-JEPA.

Supports masked forward pass: only processes patches selected by a boolean mask,
which saves computation and avoids position leak between context and target blocks.

Also supports full forward pass for downstream feature extraction.

Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.
"""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"{embed_dim} % {num_heads} != 0"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = attn @ v  # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer with masked forward capability.

    Args:
        img_size: Input image size (square).
        patch_size: Patch size for tokenization.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim ratio.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.depth = depth

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get positional embeddings for selected patch indices.

        Args:
            indices: (N_selected,) long tensor of patch indices.
        Returns:
            pos_embed: (1, N_selected, D)
        """
        return self.pos_embed[:, indices.long(), :]

    def forward_masked(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Forward pass processing only masked (visible) patches.

        Args:
            x: (B, C, H, W) input images.
            mask: (B, num_patches) boolean, True = process this patch.
            return_all_layers: If True, return list of outputs from each block.

        Returns:
            If return_all_layers=False:
                patches: (B, N_visible, D) — features for visible patches
            If return_all_layers=True:
                list of (B, N_visible, D) from each block + final norm
        """
        B = x.shape[0]

        # Patchify: (B, num_patches, D)
        patches = self.patch_embed(x)

        # Select only visible patches per sample
        # Since different samples may have different numbers of visible patches,
        # we use a list-based approach (or pad for batched processing).
        if return_all_layers:
            return self._forward_masked_multilayer(patches, mask)
        else:
            return self._forward_masked_single(patches, mask)

    def _forward_masked_single(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Single output (final norm state) for masked patches."""
        B, N, D = patches.shape

        # Collect visible patches per sample
        outputs = []
        for b in range(B):
            visible_idx = mask[b].nonzero(as_tuple=True)[0]  # (N_vis,)
            if len(visible_idx) == 0:
                # No patches selected — return empty
                outputs.append(torch.empty(0, D, device=patches.device))
                continue

            visible_patches = patches[b, visible_idx]  # (N_vis, D)
            pos_embed = self.pos_embed[0, visible_idx]  # (N_vis, D)
            tokens = visible_patches + pos_embed  # (N_vis, D)
            tokens = tokens.unsqueeze(0)  # (1, N_vis, D)

            for block in self.blocks:
                tokens = block(tokens)

            tokens = self.norm(tokens)
            outputs.append(tokens[0])  # (N_vis, D)

        return outputs

    def _forward_masked_multilayer(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[torch.Tensor]]:
        """Multi-layer output for masked patches."""
        B = patches.shape[0]
        all_layer_outputs = [[] for _ in range(self.depth + 1)]

        for b in range(B):
            visible_idx = mask[b].nonzero(as_tuple=True)[0]
            if len(visible_idx) == 0:
                for layer_idx in range(self.depth + 1):
                    all_layer_outputs[layer_idx].append(
                        torch.empty(0, self.embed_dim, device=patches.device)
                    )
                continue

            visible_patches = patches[b, visible_idx]
            pos_embed = self.pos_embed[0, visible_idx]
            tokens = visible_patches + pos_embed
            tokens = tokens.unsqueeze(0)

            all_layer_outputs[0].append(tokens[0])  # pre-block

            for i, block in enumerate(self.blocks):
                tokens = block(tokens)
                all_layer_outputs[i + 1].append(tokens[0])

        return all_layer_outputs

    def forward_full(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        Standard full forward pass (all patches + CLS token).

        Args:
            x: (B, C, H, W)
            return_all_layers: If True, return features from each block.

        Returns:
            If return_all_layers=False: (B, 1+num_patches, D)
            If return_all_layers=True: list of (B, 1+num_patches, D)
        """
        B = x.shape[0]

        patches = self.patch_embed(x)  # (B, N, D)
        patches = patches + self.pos_embed[:, :self.num_patches, :]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, patches], dim=1)  # (B, 1+N, D)

        if return_all_layers:
            outputs = [x]
            for block in self.blocks:
                x = block(x)
                outputs.append(x)
            outputs[-1] = self.norm(outputs[-1])
            return outputs

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ):
        """
        Unified forward.

        If mask is provided, does masked forward (for I-JEPA training).
        Otherwise, does full forward (for feature extraction).
        """
        if mask is not None:
            return self.forward_masked(x, mask, return_all_layers)
        return self.forward_full(x, return_all_layers)

    def get_num_patches(self) -> int:
        return self.num_patches
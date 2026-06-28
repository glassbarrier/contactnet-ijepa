"""
Multi-block mask generation for I-JEPA.

I-JEPA uses rectangular semantic blocks (not random scattered patches like MAE).
Each block covers a contiguous region — its scale and aspect ratio are sampled
from configured ranges. Blocks may overlap or be separated.

A patch belongs to a block if its center falls within the block boundaries.
"""

import torch
import random
import math
from typing import Tuple


def _sample_block_params(
    block_scale_range: Tuple[float, float],
    aspect_ratio_range: Tuple[float, float],
    image_size: int,
    patch_size: int,
) -> Tuple[float, float]:
    """
    Sample a random block's width and height in pixel space.

    Returns (block_h, block_w) in pixels.
    """
    scale = random.uniform(*block_scale_range)  # block area / image area
    aspect = random.uniform(*aspect_ratio_range)

    # Area of block in pixels
    area = scale * (image_size ** 2)

    # Derive width/height from aspect ratio
    # aspect = w / h, area = w * h  →  w = sqrt(area * aspect), h = sqrt(area / aspect)
    block_w = int(math.sqrt(area * aspect))
    block_h = int(math.sqrt(area / aspect))

    return block_h, block_w


def _block_to_patch_mask(
    block_h: int,
    block_w: int,
    image_size: int,
    patch_size: int,
    x_min: int,
    y_min: int,
) -> torch.Tensor:
    """
    Convert a pixel-space block to a patch-level boolean mask.

    Returns a 1D mask of shape (num_patches,) indicating which patches
    belong to this block.

    A patch belongs to the block if its center falls within the block rectangle.
    """
    num_patches_per_side = image_size // patch_size
    num_patches = num_patches_per_side ** 2

    mask = torch.zeros(num_patches, dtype=torch.bool)

    x_max = x_min + block_w
    y_max = y_min + block_h

    # Clamp to image bounds
    x_max = min(x_max, image_size)
    y_max = min(y_max, image_size)
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)

    patch_centers_x = torch.arange(num_patches_per_side) * patch_size + patch_size / 2
    patch_centers_y = torch.arange(num_patches_per_side) * patch_size + patch_size / 2

    grid_x, grid_y = torch.meshgrid(patch_centers_x, patch_centers_y, indexing="xy")
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    in_x = (grid_x >= x_min) & (grid_x < x_max)
    in_y = (grid_y >= y_min) & (grid_y < y_max)

    mask = in_x & in_y
    return mask


def generate_block_masks(
    image_size: int,
    patch_size: int,
    num_context_blocks: int,
    num_target_blocks: int,
    block_scale_range: Tuple[float, float] = (0.15, 0.2),
    aspect_ratio_range: Tuple[float, float] = (0.75, 1.5),
    overlap_allowed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multi-block masks for I-JEPA training.

    Args:
        image_size: Square image side length.
        patch_size: ViT patch size.
        num_context_blocks: Number of visible context blocks.
        num_target_blocks: Number of target blocks to predict.
        block_scale_range: (min, max) scale of block area relative to image area.
        aspect_ratio_range: (min, max) aspect ratio (width/height).
        overlap_allowed: Whether context and target blocks can overlap.

    Returns:
        context_mask: (num_patches,) boolean — True for patches the context encoder sees.
        target_masks: (num_target_blocks, num_patches) boolean — one mask per target block.

    The context encoder sees all context blocks combined.
    Each target block is predicted independently.
    """
    num_patches_per_side = image_size // patch_size
    num_patches = num_patches_per_side ** 2

    # Initialize masks
    context_mask = torch.zeros(num_patches, dtype=torch.bool)
    target_masks = torch.zeros(num_target_blocks, num_patches, dtype=torch.bool)

    # Sample context blocks
    for _ in range(num_context_blocks):
        block_h, block_w = _sample_block_params(
            block_scale_range, aspect_ratio_range, image_size, patch_size
        )
        x_min = random.randint(0, max(1, image_size - block_w))
        y_min = random.randint(0, max(1, image_size - block_h))

        mask = _block_to_patch_mask(block_h, block_w, image_size, patch_size, x_min, y_min)
        context_mask = context_mask | mask

    # Sample target blocks (must NOT overlap with context blocks if not allowed)
    attempts = 0
    max_attempts = 100

    for t in range(num_target_blocks):
        while attempts < max_attempts:
            block_h, block_w = _sample_block_params(
                block_scale_range, aspect_ratio_range, image_size, patch_size
            )
            x_min = random.randint(0, max(1, image_size - block_w))
            y_min = random.randint(0, max(1, image_size - block_h))

            mask = _block_to_patch_mask(block_h, block_w, image_size, patch_size, x_min, y_min)

            if overlap_allowed or not (context_mask & mask).any():
                target_masks[t] = mask
                if not overlap_allowed:
                    # Prevent future target blocks from overlapping this one too
                    context_mask = context_mask | mask
                break

            attempts += 1
        else:
            # Fallback: just use the last sampled block even if it overlaps
            target_masks[t] = mask

    return context_mask, target_masks


def masks_to_ids(
    mask: torch.Tensor,
    num_patches_per_side: int,
) -> torch.Tensor:
    """
    Convert boolean patch mask to sorted patch indices.

    Args:
        mask: (N, num_patches) boolean mask.
        num_patches_per_side: patches per side of the image grid.

    Returns:
        LongTensor of patch indices.
    """
    return mask.nonzero(as_tuple=False)[:, -1]


def generate_mask_batch(
    batch_size: int,
    image_size: int,
    patch_size: int,
    num_context_blocks: int,
    num_target_blocks: int,
    block_scale_range: Tuple[float, float] = (0.15, 0.2),
    aspect_ratio_range: Tuple[float, float] = (0.75, 1.5),
    overlap_allowed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of masks.

    Returns:
        context_masks: (B, num_patches) boolean
        target_masks_list: list of (B, num_patches) boolean, one per target block
    """
    batch_context_masks = []
    batch_target_masks = []

    for _ in range(batch_size):
        ctx_mask, tgt_masks = generate_block_masks(
            image_size, patch_size,
            num_context_blocks, num_target_blocks,
            block_scale_range, aspect_ratio_range,
            overlap_allowed,
        )
        batch_context_masks.append(ctx_mask)
        batch_target_masks.append(tgt_masks)

    context_masks = torch.stack(batch_context_masks, dim=0)  # (B, N_patches)

    # target_masks: (num_target_blocks, B, N_patches)
    target_masks = torch.stack(batch_target_masks, dim=0)

    return context_masks, target_masks
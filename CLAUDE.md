# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

I-JEPA (Image-based Joint-Embedding Predictive Architecture) for contact network
component anomaly detection. Uses self-supervised pretraining on normal component
images, then downstream anomaly detection via patch distribution modeling (PaDiM-style).

**Dataset**: 12 contact network component categories in MVTec-AD format.
Data lives on a remote server (`~/UCAD/mvtec2d/`), NOT in this repo.
All paths are configured via YAML files — never hardcoded.

**Key references**:
- I-JEPA: Assran et al., arXiv:2301.08243
- PaDiM: Defard et al., arXiv:2011.08785

## Architecture

```
I-JEPA Pretraining:
  Context Encoder (ViT)  →  Predictor (narrower Transformer)  →  Target Encoder (EMA, stop-grad)
      sees context blocks       maps ctx→tgt in latent space        sees target blocks (ground truth)
  Loss: L2 between predictor output and target encoder output

Downstream Anomaly Detection:
  Pretrained Context Encoder  →  Patch Features (multi-layer)  →  Per-patch Gaussian (Mahalanobis distance)
```

## Project Structure

```
ijepa/
├── config/
│   ├── ijepa_pretrain.yaml      # Pretraining hyperparams, data paths
│   └── anomaly_detection.yaml   # Downstream evaluation config
├── src/
│   ├── data/                    # Dataset loading, masks, augmentations
│   ├── models/                  # ViT backbone, I-JEPA predictor, full model
│   ├── training/                # Trainer loop, EMA/LR schedulers
│   ├── features/                # Feature extraction + bank building
│   ├── anomaly/                 # PaDiM detector + evaluator
│   └── utils/                   # Config loader, metrics
├── scripts/
│   ├── pretrain.py              # I-JEPA pretraining entry point
│   ├── extract_features.py      # Feature extraction from checkpoint
│   └── anomaly_detect.py        # Anomaly detection evaluation
├── requirements.txt
└── .gitignore
```

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# I-JEPA pretraining
python scripts/pretrain.py --config config/ijepa_pretrain.yaml

# Override config values from CLI
python scripts/pretrain.py --config config/ijepa_pretrain.yaml \
    -o data.root=/path/to/dataset data.batch_size=64

# Extract features from pretrained checkpoint
python scripts/extract_features.py --config config/anomaly_detection.yaml \
    -o features.checkpoint=checkpoints/ijepa_best.pth

# Run anomaly detection evaluation
python scripts/anomaly_detect.py --config config/anomaly_detection.yaml \
    -o features.checkpoint=checkpoints/ijepa_best.pth
```

## Data Format

The dataset follows MVTec-AD structure:
```
<root>/
  <category>/
    train/good/          # Normal images for pretraining
    test/good/           # Normal test images
    test/<defect_type>/  # Anomalous test images
    ground_truth/<defect_type>/  # Pixel-level anomaly masks
```

For pretraining, ALL `train/good` images from ALL categories are pooled.
For anomaly detection, each category is evaluated separately.

Special case: `ear_croped/train/二分类good` is ignored; only `ear_croped/train/good` is used.

## Configuration System

All paths and hyperparams come from YAML config files. CLI overrides use `-o key=value`:
- `data.root` — dataset path (REQUIRED: set to your server path)
- `training.epochs` — number of pretraining epochs
- `features.checkpoint` — checkpoint path for downstream tasks

The config loader (`src/utils/config.py`) supports dot-separated nested keys.

## Key Design Decisions

1. **Only normal images for pretraining**: I-JEPA sees only `train/good` during
   pretraining. This ensures the model learns "normal" representations, making
   anomalies easier to detect.

2. **EMA target encoder**: The target encoder is never trained by gradients.
   It is updated via exponential moving average of the context encoder.

3. **Multi-block masks**: I-JEPA uses rectangular semantic blocks (not scattered
   patches like MAE). Block scales and aspect ratios are configurable.

4. **Multi-layer features for anomaly detection**: Features from multiple ViT
   layers are concatenated (PaDiM-style) for richer patch representations.

5. **Per-category evaluation**: Anomaly detection is evaluated separately per
   component category, since normal variation differs by component type.

## Typical Workflow

1. Edit `config/ijepa_pretrain.yaml` → set `data.root` to your dataset path
2. Run `scripts/pretrain.py` for self-supervised pretraining
3. Run `scripts/anomaly_detect.py` to evaluate anomaly detection
   (this auto-runs feature extraction if no bank exists)
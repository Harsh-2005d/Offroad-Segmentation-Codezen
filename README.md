# Offroad Semantic Scene Segmentation

**PyTorch · DINOv2 · OpenCV · Albumentations**  
*Hackathon project — Duality AI Offroad Challenge (Team ORCA)*

---

<p align="center">
  <img src="docs/model_architecture.jpeg" alt="Model Architecture" width="900"/>
</p>

## Overview

Hybrid Transformer–CNN semantic segmentation model for off-road desert environments, trained on synthetic data from Duality AI's Falcon platform. Segments 10 terrain classes including rocks, vegetation, and navigable ground across challenging low-contrast scenes.

---

## Architecture

Fused **multi-depth DINOv2** (ViT) features with a manually constructed **CNN feature pyramid neck** to recover the spatial hierarchy that transformers lack natively.

- Features extracted from early, mid, and final transformer blocks — capturing texture, object structure, and global semantics respectively
- Pyramid levels at 9×9 → 72×72 built from the 36×36 token grid
- Per-scale segmentation heads with **deep supervision** improve rare-class gradient flow and boundary sharpness

This bypasses the need for a conventional decoder by constructing spatial inductive bias explicitly from transformer patch tokens.

---

## Loss & Training

Compound loss — **CrossEntropy + Dice + Focal** — to handle severe class imbalance across sparse terrain features (e.g., logs, flowers, rocks vs. dominant sky/landscape).

| Hyperparameter | Value |
|---|---|
| Input size | 512×512 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |

Augmentation pipeline (Albumentations): random crop, horizontal flip, brightness/contrast jitter, ImageNet normalization.

---

## Results

| Metric | Value |
|---|---|
| Mean IoU (mIoU) | ~0.70 |
| Pixel Accuracy | ~0.85 |
| Inference Latency | ~4 ms/image |
| Throughput | ~200 FPS |

---

## Engineering

- End-to-end training, evaluation, and checkpointing pipeline
- Visualization tooling: segmentation overlays, per-class IoU plots, confusion matrices, input/GT/prediction comparisons
- Latency benchmarking for deployment-readiness assessment

```bash
python train.py          # Training
python test.py           # Evaluation
python visualize_segmentation.py  # Qualitative results
```

---

## Semantic Classes

| ID | Class |
|---|---|
| 100 | Trees |
| 200 | Lush Bushes |
| 300 | Dry Grass |
| 500 | Dry Bushes |
| 550 | Ground Clutter *(walkable path — highlighted)* |
| 600 | Flowers |
| 700 | Logs |
| 800 | Rocks |
| 7100 | Landscape |
| 10000 | Sky |
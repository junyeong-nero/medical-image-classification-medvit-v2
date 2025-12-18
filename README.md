# Brain Tumor Classification with MedViT-v2

A deep learning project for classifying brain tumors from MRI images using **MedViT-v2** (Medical Vision Transformer).

## Overview

This project implements an automated brain tumor classification system using MedViT-v2, a hybrid transformer architecture optimized for medical imaging. It supports 4 tumor categories: **Glioma, Meningioma, Pituitary, and No Tumor**.

### Key Features

- **Hybrid Architecture**: MedViT-v2 combines CNNs and Transformers for optimal performance.
- **Multi-Device Support**: Automatic detection for CUDA (NVIDIA), MPS (Apple Silicon), and CPU.
- **Flexible Data**: Supports any HuggingFace image classification dataset.
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC, etc.

## Dataset

**Source**: [PranomVignesh/MRI-Images-of-Brain-Tumor](https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor)
**Input**: 224×224 MRI images

## Model Architecture

**MedViT-v2** uses a hybrid approach with Convolutional Stems and Hybrid Processing Blocks (combining Neighborhood Attention and Global Attention).

**Available Sizes**: `MedViT_tiny` (default), `MedViT_small`, `MedViT_base`, `MedViT_large`.

## Installation

**Requirements**: Python 3.13+, `uv` (recommended) or `pip`.

```bash
# Clone and install dependencies
git clone <repository-url>
cd medical-test
uv sync  # or: pip install -r requirements.txt
source .venv/bin/activate
```

## Usage

### Quick Start
Train a `MedViT_tiny` model with default settings:
```bash
bash scripts/train.sh
```

### Advanced Training
Customize parameters like model size, batch size, and dataset:

```bash
uv run python src/train.py \
    --model_name MedViT_small \
    --dataset PranomVignesh/MRI-Images-of-Brain-Tumor \
    --batch_size 16 \
    --epochs 30 \
    --pretrained True
```

**Common Arguments**:
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | Architecture (`MedViT_tiny`, `resnet50`, etc.) | `MedViT_tiny` |
| `--dataset` | HuggingFace dataset name | - |
| `--batch_size` | Training batch size | `32` |
| `--epochs` | Number of epochs | `10` |
| `--pretrained` | Use pretrained weights | `False` |

## Project Structure

- `src/models/`: MedViT-v2 architecture definitions.
- `src/train.py`: Main training script.
- `scripts/`: Helper shell scripts for training.
- `weights/`: Stores trained models and checkpoints.

## Citation & License

If you use this work, please cite the original MedViT paper.
License information: [Add License]
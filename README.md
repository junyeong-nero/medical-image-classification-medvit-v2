# Brain Tumor Classification with MedViT-v2

A deep learning project for classifying brain tumors from MRI images using **MedViT-v2** (Medical Vision Transformer), a state-of-the-art hybrid transformer architecture specifically designed for medical image analysis.

## Overview

This project implements an automated brain tumor classification system that can identify different types of brain tumors from MRI scans. The model leverages MedViT-v2, which combines the strengths of Vision Transformers with convolutional neural networks, optimized for medical imaging tasks.

### Key Features

- **MedViT-v2 Architecture**: Hybrid vision transformer with local and global feature processing
- **Multi-class Classification**: Supports 4 tumor categories
- **Pretrained Weights**: Optional pretrained models for transfer learning
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, specificity, and AUC-ROC
- **Easy Training**: Simple command-line interface with customizable hyperparameters

## Dataset

**Source**: [PranomVignesh/MRI-Images-of-Brain-Tumor](https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor)

**Input**: Brain MRI images (224×224 pixels)

**Classes** (4 categories):
1. **Glioma** - Tumors that arise from glial cells
2. **Meningioma** - Tumors that form on membranes covering the brain and spinal cord
3. **Pituitary** - Tumors in the pituitary gland
4. **No Tumor** - Healthy brain scans without tumors

## Model Architecture

### MedViT-v2 (Medical Vision Transformer)

MedViT-v2 is a hybrid architecture that combines convolutional operations with transformer-based attention mechanisms, specifically optimized for medical imaging:

**Components**:

1. **Convolutional Stem** (4 layers)
   - Downsamples input images from 224×224 to 56×56
   - Extracts low-level features

2. **Hybrid Processing Blocks** (LFP + GFP)
   - **LFP (Local Feature Processing)**: Uses Neighborhood Attention (NATTEN) for capturing local spatial relationships with depthwise convolutions
   - **GFP (Global Feature Processing)**: Combines E-MHSA (Efficient Multi-Head Self Attention) with MHCA (Multi-Head Convolutional Attention), and employs KAN (Kolmogorov-Arnold Networks) for non-linear transformations

3. **Classification Head**
   - Global average pooling
   - Linear projection to 4 classes

**Available Model Sizes**:
- `MedViT_tiny` - Lightweight, faster training
- `MedViT_small` - Balanced performance
- `MedViT_base` - Higher capacity
- `MedViT_large` - Maximum performance

**Key Technologies**:
- **NATTEN**: Neighborhood Attention for efficient local attention
- **FasterKAN**: Fast Kolmogorov-Arnold Network implementation
- **ECA/SE Modules**: Efficient channel and squeeze-excitation attention

## Requirements

- **Python**: 3.13 or higher
- **Package Manager**: uv (recommended) or pip
- **Hardware**: CUDA-capable GPU recommended (CPU supported)

### Dependencies

```toml
torch>=2.9.1
torchvision>=0.24.1
transformers>=4.57.3
timm>=1.0.22
natten>=0.21.1
datasets>=4.4.1
scikit-learn>=1.8.0
matplotlib>=3.10.8
numpy>=2.3.5
pandas>=2.3.3
einops>=0.8.1
fastcan>=0.5.0
medmnist>=3.0.2
```

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd medical-test

# Install dependencies
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Training

The simplest way to start training:

```bash
bash scripts/train.sh
```

This will train a `MedViT_tiny` model on the brain tumor dataset with default parameters.

### Custom Training

Train with custom hyperparameters:

```bash
uv run python src/train.py \
    --model_name MedViT_tiny \
    --dataset PranomVignesh/MRI-Images-of-Brain-Tumor \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 50 \
    --pretrained False
```

### Training with Pretrained Weights

Use pretrained MedViT weights for transfer learning:

```bash
uv run python src/train.py \
    --model_name MedViT_small \
    --dataset PranomVignesh/MRI-Images-of-Brain-Tumor \
    --batch_size 16 \
    --lr 0.00005 \
    --epochs 30 \
    --pretrained True \
    --checkpoint_path ./checkpoint/MedViT_small.pth
```

The pretrained weights will be automatically downloaded if not present.

### Training Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--model_name` | Model architecture | `MedViT_tiny` | `MedViT_tiny`, `MedViT_small`, `MedViT_base`, `MedViT_large`, or any `timm` model |
| `--dataset` | Dataset name | - | HuggingFace dataset name or MedMNIST dataset |
| `--batch_size` | Training batch size | `32` | Integer (validation uses 2×batch_size) |
| `--lr` | Learning rate | `0.0001` | Float (typical: 0.00001-0.001) |
| `--epochs` | Number of epochs | `10` | Integer |
| `--pretrained` | Use pretrained weights | `False` | `True` or `False` |
| `--checkpoint_path` | Path to checkpoint | `./checkpoint/MedViT_tiny.pth` | String |

### Dataset Inspection

Explore the dataset structure and visualize samples:

```bash
uv run python src/inspect_dataset.py
```

## Training Pipeline

### Data Preprocessing

- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Training Augmentation**:
  - Resize to 224×224
  - Random horizontal flip
- **Validation**: Resize to 224×224 only

### Training Configuration

- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing with Warm Restarts
- **Loss Function**: Cross-Entropy Loss
- **Evaluation Metrics**:
  - Accuracy
  - Precision (macro-averaged)
  - Recall (macro-averaged)
  - F1-Score (macro-averaged)
  - Specificity
  - AUC-ROC (One-vs-Rest multi-class)
  - Confusion Matrix

### Model Checkpoints

Checkpoints are automatically saved when validation accuracy improves. Each checkpoint includes:
- Model state dictionary
- Optimizer state
- Scheduler state
- Best validation accuracy
- Current epoch number

**Checkpoint naming**: `{model_name}_{dataset_name}.pth`

## Project Structure

```
medical-test/
├── src/
│   ├── models/
│   │   ├── MedViT.py          # MedViT-v2 architecture implementation
│   │   ├── fasterkan.py       # FasterKAN module
│   │   └── __init__.py
│   ├── train.py               # Training script with evaluation
│   ├── dataset_builder.py     # Dataset loading and preprocessing
│   ├── inspect_dataset.py     # Dataset inspection utility
│   └── utils.py               # Helper functions (download, etc.)
├── scripts/
│   └── train.sh               # Training shell script
├── checkpoint/                # Model checkpoints (created during training)
├── main.py                    # Entry point
├── pyproject.toml            # Project dependencies
└── README.md
```

## Examples

### Example 1: Quick Training with Default Settings

```bash
# Train MedViT_tiny from scratch
bash scripts/train.sh
```

### Example 2: Training a Larger Model

```bash
# Train MedViT_base with larger batch size
uv run python src/train.py \
    --model_name MedViT_base \
    --dataset PranomVignesh/MRI-Images-of-Brain-Tumor \
    --batch_size 16 \
    --lr 0.0001 \
    --epochs 100
```

### Example 3: Fine-tuning with Pretrained Weights

```bash
# Fine-tune pretrained MedViT_small
uv run python src/train.py \
    --model_name MedViT_small \
    --dataset PranomVignesh/MRI-Images-of-Brain-Tumor \
    --batch_size 24 \
    --lr 0.00005 \
    --epochs 50 \
    --pretrained True
```

### Example 4: Training on Different Datasets

```bash
# Train on a MedMNIST dataset
uv run python src/train.py \
    --model_name MedViT_tiny \
    --dataset pathmnist \
    --batch_size 64 \
    --lr 0.0001 \
    --epochs 30
```

### Example 5: Using timm Models

```bash
# Train with a ResNet from timm library
uv run python src/train.py \
    --model_name resnet50 \
    --dataset PranomVignesh/MRI-Images-of-Brain-Tumor \
    --batch_size 32 \
    --lr 0.001 \
    --epochs 50
```

## Performance Tips

1. **Batch Size**: Start with smaller batches if running out of GPU memory
2. **Learning Rate**: Use lower learning rates (0.00005) when fine-tuning pretrained models
3. **Model Size**: Start with `MedViT_tiny` for quick experimentation, then scale up
4. **Epochs**: Medical imaging typically requires 50-100 epochs for convergence
5. **Data Augmentation**: The pipeline includes horizontal flips; consider adding more augmentations for better generalization

## Citation

If you use this code or MedViT architecture in your research, please cite the original MedViT paper:

```bibtex
@article{medvit2024,
  title={MedViT: A robust vision transformer for medical image classification},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

[Add your license information here]

## Acknowledgments

- Dataset: [PranomVignesh](https://huggingface.co/PranomVignesh) for the MRI brain tumor dataset
- Architecture: MedViT-v2 implementation based on the original paper
- Libraries: PyTorch, Hugging Face, NATTEN, timm

## Contact

[Add your contact information here]

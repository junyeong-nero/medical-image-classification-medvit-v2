# Weights Directory

This directory stores all model weights for the project.

## Directory Structure

```
weights/
├── pretrained/          # Downloaded pretrained model weights
│   ├── MedViT_tiny.pth
│   ├── MedViT_small.pth
│   ├── MedViT_base.pth
│   └── MedViT_large.pth
├── {model}_{dataset}.pth  # Your trained model checkpoints
└── README.md
```

## Pretrained Weights

Pretrained weights are automatically downloaded to `pretrained/` when you use `--pretrained True`:

```bash
python src/train.py \
    --model_name MedViT_small \
    --pretrained True
```

The script will:
1. Check if `weights/pretrained/MedViT_small.pth` exists
2. If not, download from the official source
3. Load the weights for transfer learning

## Trained Model Checkpoints

Your trained models are saved directly in the `weights/` directory with the naming format:
- `{model_name}_{dataset_name}.pth`

For example:
- `MedViT_tiny_PranomVignesh_MRI-Images-of-Brain-Tumor.pth`
- `MedViT_small_pathmnist.pth`

Each checkpoint includes:
- Model state dictionary
- Optimizer state
- Learning rate scheduler state
- Best validation accuracy
- Current epoch number

## Using Custom Checkpoint Paths

You can specify a custom checkpoint path:

```bash
python src/train.py \
    --model_name MedViT_small \
    --pretrained True \
    --checkpoint_path /path/to/your/checkpoint.pth
```

## Security Note

Only load checkpoints from trusted sources. The code uses `weights_only=False` to support models with NumPy objects, which can execute arbitrary code if the checkpoint is malicious.

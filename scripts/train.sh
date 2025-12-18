#!/bin/bash

# Ensure script stops on error
set -e

# Training configuration for Brain Tumor Classification
# Dataset: PranomVignesh/MRI-Images-of-Brain-Tumor
# Model: MedViT_tiny (starting with the smallest model)

echo "================================================"
echo "Brain Tumor Classification Training"
echo "================================================"
echo "Dataset: PranomVignesh/MRI-Images-of-Brain-Tumor"
echo "Model: MedViT_tiny"
echo "================================================"
echo ""

# Run training using uv
uv run python src/train.py \
    --model_name MedViT_tiny \
    --dataset PranomVignesh/MRI-Images-of-Brain-Tumor \
    --batch_size 8 \
    --lr 0.0001 \
    --epochs 50 \
    --pretrained False \
    --image_column image \
    --label_column label

echo ""
echo "================================================"
echo "Training completed!"
echo "================================================"

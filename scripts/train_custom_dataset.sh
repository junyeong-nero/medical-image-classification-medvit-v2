#!/bin/bash

# Ensure script stops on error
set -e

# Example: Training with a Custom HuggingFace Dataset
# This script demonstrates how to use different image/label column names
#
# IMPORTANT: Replace the following with your actual dataset information:
# - DATASET_NAME: Your HuggingFace dataset name (e.g., "username/dataset-name")
# - IMAGE_COLUMN: Column name containing images (e.g., "img", "image", "picture")
# - LABEL_COLUMN: Column name containing labels (e.g., "label", "class", "category")

DATASET_NAME="PranomVignesh/MRI-Images-of-Brain-Tumor"  # Replace with your dataset
IMAGE_COLUMN="image"  # Replace with your image column name
LABEL_COLUMN="label"  # Replace with your label column name

echo "================================================"
echo "Custom Dataset Training"
echo "================================================"
echo "Dataset: $DATASET_NAME"
echo "Image Column: $IMAGE_COLUMN"
echo "Label Column: $LABEL_COLUMN"
echo "Model: MedViT_tiny"
echo "================================================"
echo ""

# Run training using uv
uv run python src/train.py \
    --model_name MedViT_tiny \
    --dataset "$DATASET_NAME" \
    --batch_size 8 \
    --lr 0.0001 \
    --epochs 50 \
    --pretrained False \
    --image_column "$IMAGE_COLUMN" \
    --label_column "$LABEL_COLUMN"

echo ""
echo "================================================"
echo "Training completed!"
echo "================================================"

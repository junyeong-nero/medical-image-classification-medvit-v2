#!/bin/bash

# Ensure script stops on error
set -e

# Example: Training on Leukemia Blood Cell Dataset (C-NMC 2019)
# Binary classification: ALL (Acute Lymphoblastic Leukemia) vs Normal
#
# Dataset Info:
# - Source: dwb2023/cnmc-leukemia-2019
# - Samples: 10,661 microscopic blood cell images
# - Task: Binary classification (cancer vs healthy)
# - Modality: Microscopy images of white blood cells

DATASET_NAME="dwb2023/cnmc-leukemia-2019"
IMAGE_COLUMN="image"
LABEL_COLUMN="label"  # "cancer" or "healthy"

echo "================================================"
echo "Leukemia Blood Cell Classification Training"
echo "================================================"
echo "Dataset: $DATASET_NAME (C-NMC 2019)"
echo "Task: ALL (Acute Lymphoblastic Leukemia) Detection"
echo "Image Column: $IMAGE_COLUMN"
echo "Label Column: $LABEL_COLUMN"
echo "Model: MedViT_tiny"
echo ""
echo "Classes:"
echo "  - cancer (ALL cells)"
echo "  - healthy (normal cells)"
echo "================================================"
echo ""

# Run training using uv
uv run python src/train.py \
    --model_name MedViT_tiny \
    --dataset "$DATASET_NAME" \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 30 \
    --pretrained True \
    --image_column "$IMAGE_COLUMN" \
    --label_column "$LABEL_COLUMN"

echo ""
echo "================================================"
echo "Training completed!"
echo "================================================"

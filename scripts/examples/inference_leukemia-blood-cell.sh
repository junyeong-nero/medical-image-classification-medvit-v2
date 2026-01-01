#!/bin/bash

# Ensure script stops on error
set -e

# Example: Inference on Leukemia Blood Cell Dataset (C-NMC 2019)
# Evaluate ALL (Acute Lymphoblastic Leukemia) detection performance

DATASET_NAME="dwb2023/cnmc-leukemia-2019"
MODEL_WEIGHTS="weights/MedViT_tiny_dwb2023_cnmc-leukemia-2019_best_weights.pth"
SPLIT="train"  # This dataset only has a train split - consider custom splitting
IMAGE_COLUMN="image"
LABEL_COLUMN="label"  # "cancer" or "healthy"

echo "================================================"
echo "Leukemia Blood Cell Inference"
echo "================================================"
echo "Dataset: $DATASET_NAME (C-NMC 2019)"
echo "Task: ALL (Acute Lymphoblastic Leukemia) Detection"
echo "Split: $SPLIT"
echo "Model: MedViT_tiny"
echo "Weights: $MODEL_WEIGHTS"
echo ""
echo "Binary Classification Metrics:"
echo "  - Sensitivity (True Positive Rate for cancer)"
echo "  - Specificity (True Negative Rate for healthy)"
echo "  - AUC-ROC"
echo "================================================"
echo ""

# Check if weights exist
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "Warning: Model weights not found at $MODEL_WEIGHTS"
    echo "Please train the model first using: ./train_leukemia-blood-cell.sh"
    exit 1
fi

# Run inference using uv
uv run python src/inference.py \
    --model_name MedViT_tiny \
    --model_weights "$MODEL_WEIGHTS" \
    --dataset "$DATASET_NAME" \
    --split "$SPLIT" \
    --batch_size 32 \
    --image_column "$IMAGE_COLUMN" \
    --label_column "$LABEL_COLUMN" \
    --evaluate True \
    --save_results True

echo ""
echo "================================================"
echo "Inference completed!"
echo "Results saved to: results/"
echo ""
echo "For binary classification, focus on:"
echo "  - Sensitivity/Recall (detecting cancer cases)"
echo "  - Specificity (avoiding false positives)"
echo "  - AUC-ROC (overall discrimination)"
echo "================================================"

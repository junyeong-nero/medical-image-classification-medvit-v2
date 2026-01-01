#!/bin/bash

# Ensure script stops on error
set -e

# Example: Inference on Imbalanced Skin Cancer Dataset (HAM10000)
# Evaluate model performance on rare classes like dermatofibroma

DATASET_NAME="marmal88/skin_cancer"
MODEL_WEIGHTS="weights/MedViT_tiny_marmal88_skin_cancer_best_weights.pth"
SPLIT="test"
IMAGE_COLUMN="image"
LABEL_COLUMN="dx"  # Diagnosis column contains class labels

echo "================================================"
echo "Imbalanced Skin Cancer Inference"
echo "================================================"
echo "Dataset: $DATASET_NAME"
echo "Split: $SPLIT"
echo "Model: MedViT_tiny"
echo "Weights: $MODEL_WEIGHTS"
echo ""
echo "Pay attention to per-class metrics for rare classes:"
echo "  - dermatofibroma (df)"
echo "  - vascular lesions (vasc)"
echo "  - actinic keratoses (akiec)"
echo "================================================"
echo ""

# Run inference using uv
uv run python src/inference.py \
    --model_name MedViT_tiny \
    --model_weights "$MODEL_WEIGHTS" \
    --dataset "$DATASET_NAME" \
    --split "$SPLIT" \
    --batch_size 16 \
    --image_column "$IMAGE_COLUMN" \
    --label_column "$LABEL_COLUMN" \
    --evaluate True \
    --save_results True

echo ""
echo "================================================"
echo "Inference completed!"
echo "Results saved to: results/"
echo "Check per-class F1 scores for rare disease classes!"
echo "================================================"

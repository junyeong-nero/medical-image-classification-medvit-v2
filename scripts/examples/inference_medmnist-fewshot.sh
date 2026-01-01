#!/bin/bash

# Ensure script stops on error
set -e

# Example: Inference on MedMNIST Small Datasets
# Evaluate few-shot learning performance
#
# Usage: 
#   ./inference_medmnist-fewshot.sh              # Default: breastmnist
#   ./inference_medmnist-fewshot.sh retinamnist  # Use retinamnist

# Select dataset (default: breastmnist)
MEDMNIST_DATASET="${1:-breastmnist}"

case "$MEDMNIST_DATASET" in
    "breastmnist")
        DESCRIPTION="Breast Ultrasound (780 samples, 2 classes)"
        ;;
    "retinamnist")
        DESCRIPTION="Retinal OCT (1,600 samples, 5 classes)"
        ;;
    "dermamnist")
        DESCRIPTION="Skin Lesion (10,015 samples, 7 classes)"
        ;;
    "bloodmnist")
        DESCRIPTION="Blood Cell (17,092 samples, 8 classes)"
        ;;
    *)
        echo "Unknown dataset: $MEDMNIST_DATASET"
        echo "Available: breastmnist, retinamnist, dermamnist, bloodmnist"
        exit 1
        ;;
esac

MODEL_WEIGHTS="weights/MedViT_tiny_${MEDMNIST_DATASET}_best_weights.pth"

echo "================================================"
echo "MedMNIST Few-shot Inference"
echo "================================================"
echo "Dataset: $MEDMNIST_DATASET"
echo "Description: $DESCRIPTION"
echo "Model: MedViT_tiny"
echo "Weights: $MODEL_WEIGHTS"
echo ""
echo "Evaluating few-shot learning performance..."
echo "================================================"
echo ""

# Check if weights exist
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "Warning: Model weights not found at $MODEL_WEIGHTS"
    echo "Please train the model first using: ./train_medmnist-fewshot.sh $MEDMNIST_DATASET"
    exit 1
fi

uv run python src/inference.py \
    --model_name MedViT_tiny \
    --model_weights "$MODEL_WEIGHTS" \
    --dataset "$MEDMNIST_DATASET" \
    --split test \
    --batch_size 32 \
    --evaluate True \
    --save_results True

echo ""
echo "================================================"
echo "Few-shot inference completed!"
echo "Dataset: $MEDMNIST_DATASET"
echo "Results saved to: results/"
echo "================================================"

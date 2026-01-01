#!/bin/bash

# Ensure script stops on error
set -e

# Example: Few-shot Training on MedMNIST Small Datasets
# Tests model performance with limited training data
#
# Small MedMNIST Datasets (good for few-shot testing):
# - BreastMNIST: 780 samples, 2 classes (breast ultrasound)
# - RetinaMNIST: 1,600 samples, 5 classes (retinal OCT)
#
# Usage: 
#   ./train_medmnist-fewshot.sh              # Default: breastmnist
#   ./train_medmnist-fewshot.sh retinamnist  # Use retinamnist

# Select dataset (default: breastmnist for smallest dataset)
MEDMNIST_DATASET="${1:-breastmnist}"

case "$MEDMNIST_DATASET" in
    "breastmnist")
        DESCRIPTION="Breast Ultrasound (780 samples, 2 classes)"
        NUM_CLASSES=2
        ;;
    "retinamnist")
        DESCRIPTION="Retinal OCT (1,600 samples, 5 classes)"
        NUM_CLASSES=5
        ;;
    "dermamnist")
        DESCRIPTION="Skin Lesion (10,015 samples, 7 classes)"
        NUM_CLASSES=7
        ;;
    "bloodmnist")
        DESCRIPTION="Blood Cell (17,092 samples, 8 classes)"
        NUM_CLASSES=8
        ;;
    *)
        echo "Unknown dataset: $MEDMNIST_DATASET"
        echo "Available: breastmnist, retinamnist, dermamnist, bloodmnist"
        exit 1
        ;;
esac

echo "================================================"
echo "MedMNIST Few-shot Training"
echo "================================================"
echo "Dataset: $MEDMNIST_DATASET"
echo "Description: $DESCRIPTION"
echo "Model: MedViT_tiny"
echo ""
echo "This tests model performance with LIMITED training data."
echo "================================================"
echo ""

uv run python src/train.py \
    --model_name MedViT_tiny \
    --dataset "$MEDMNIST_DATASET" \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 50 \
    --pretrained True

echo ""
echo "================================================"
echo "Few-shot training completed!"
echo "Dataset: $MEDMNIST_DATASET"
echo "================================================"

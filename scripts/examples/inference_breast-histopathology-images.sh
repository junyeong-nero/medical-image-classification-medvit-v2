#!/bin/bash

# Ensure script stops on error
set -e

# Example: Inference with a Custom HuggingFace Dataset
# This script demonstrates how to run inference on a trained model
#
# IMPORTANT: Replace the following with your actual information:
# - DATASET_NAME: Your HuggingFace dataset name (e.g., "username/dataset-name")
# - MODEL_WEIGHTS: Path to your trained model weights
# - SPLIT: Dataset split to run inference on (e.g., "test", "validation")
# - IMAGE_COLUMN: Column name containing images (e.g., "img", "image", "picture")
# - LABEL_COLUMN: Column name containing labels (e.g., "label", "class", "category")

DATASET_NAME="junyeong-nero/mini-breast-histopathology-images"  # Replace with your dataset
MODEL_WEIGHTS="weights/MedViT_tiny_junyeong-nero_mini-breast-histopathology-images.pth"  # Path to trained weights
SPLIT="test"  # Dataset split: test, validation, train
IMAGE_COLUMN="image"  # Replace with your image column name
LABEL_COLUMN="label"  # Replace with your label column name

echo "================================================"
echo "Model Inference on Custom Dataset"
echo "================================================"
echo "Dataset: $DATASET_NAME"
echo "Split: $SPLIT"
echo "Model: MedViT_tiny"
echo "Weights: $MODEL_WEIGHTS"
echo "Image Column: $IMAGE_COLUMN"
echo "Label Column: $LABEL_COLUMN"
echo "================================================"
echo ""

# Run inference using uv
uv run python src/inference.py \
    --model_name MedViT_tiny \
    --model_weights "$MODEL_WEIGHTS" \
    --dataset "$DATASET_NAME" \
    --split "$SPLIT" \
    --batch_size 8 \
    --image_column "$IMAGE_COLUMN" \
    --label_column "$LABEL_COLUMN" \
    --evaluate True \
    --save_results True

echo ""
echo "================================================"
echo "Inference completed!"
echo "Results saved to: results/"
echo "================================================"

#!/bin/bash

# Ensure script stops on error
set -e

# Example: Training on Imbalanced Skin Cancer Dataset (HAM10000)
# This dataset has highly imbalanced classes - ideal for testing rare disease scenarios
#
# Class Distribution (approximate):
# - nv (melanocytic nevi): ~6,700 samples (majority)
# - mel (melanoma): ~1,100 samples
# - bkl (benign keratosis): ~1,100 samples
# - bcc (basal cell carcinoma): ~500 samples
# - akiec (actinic keratoses): ~300 samples
# - vasc (vascular lesions): ~140 samples
# - df (dermatofibroma): ~115 samples (RARE!)

DATASET_NAME="marmal88/skin_cancer"
IMAGE_COLUMN="image"
LABEL_COLUMN="dx"  # Diagnosis column contains class labels

echo "================================================"
echo "Imbalanced Skin Cancer Dataset Training"
echo "================================================"
echo "Dataset: $DATASET_NAME"
echo "Image Column: $IMAGE_COLUMN"
echo "Label Column: $LABEL_COLUMN"
echo "Model: MedViT_tiny"
echo ""
echo "This dataset is IMBALANCED with rare classes like:"
echo "  - dermatofibroma (df): ~115 samples"
echo "  - vascular lesions (vasc): ~140 samples"
echo "================================================"
echo ""

# Run training using uv
uv run python src/train.py \
    --model_name MedViT_tiny \
    --dataset "$DATASET_NAME" \
    --batch_size 4 \
    --lr 0.0001 \
    --epochs 30 \
    --pretrained True \
    --image_column "$IMAGE_COLUMN" \
    --label_column "$LABEL_COLUMN"

echo ""
echo "================================================"
echo "Training completed!"
echo "================================================"

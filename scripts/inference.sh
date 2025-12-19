#!/bin/bash

# Inference script for brain tumor classification
# This script runs inference using the trained MedViT_tiny model on the test split
# Use --evaluate True to compute comprehensive evaluation metrics

uv run src/inference.py \
    --model_name MedViT_tiny \
    --model_weights weights/MedViT_tiny_PranomVignesh_MRI-Images-of-Brain-Tumor.pth \
    --dataset PranomVignesh/MRI-Images-of-Brain-Tumor \
    --split test \
    --batch_size 8 \
    --evaluate True \
    --save_results True \
    # --push_to_hub junyeong-nero/brain-tumor-predictions-test
    # --private_hub False

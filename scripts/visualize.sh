#!/bin/bash

# Ensure script stops on error
set -e

# Attention Map Visualization for MedViT
# Visualizes attention patterns from trained models

echo "================================================"
echo "MedViT Attention Map Visualization"
echo "================================================"

# Default configuration
MODEL_NAME="${MODEL_NAME:-MedViT_tiny}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./checkpoint/MedViT_tiny_PranomVignesh_MRI-Images-of-Brain-Tumor.pth}"
NUM_CLASSES="${NUM_CLASSES:-4}"
IMAGE_PATH="${IMAGE_PATH:-assets/brain-tumor-001.jpg}"
OUTPUT_PATH="${OUTPUT_PATH:-./brain-tumor-001-attention-visualization.png}"
CMAP="${CMAP:-jet}"
ALPHA="${ALPHA:-0.5}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image_path)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num_classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --cmap)
            CMAP="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --no-show)
            NO_SHOW="--no-show"
            shift
            ;;
        -h|--help)
            echo ""
            echo "Usage: ./scripts/visualize.sh --image_path <path> [options]"
            echo ""
            echo "Required:"
            echo "  --image_path        Path to input image"
            echo ""
            echo "Optional:"
            echo "  --checkpoint_path   Path to model checkpoint (default: ./checkpoint/MedViT_tiny_brain_tumor.pth)"
            echo "  --model_name        Model architecture: MedViT_tiny, MedViT_small, MedViT_base, MedViT_large"
            echo "                      (default: MedViT_tiny)"
            echo "  --num_classes       Number of output classes (default: 4)"
            echo "  --output_path       Path to save visualization (default: ./attention_visualization.png)"
            echo "  --cmap              Colormap for attention overlay (default: jet)"
            echo "  --alpha             Transparency of attention overlay, 0-1 (default: 0.5)"
            echo "  --no-show           Do not display visualization window"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic usage"
            echo "  ./scripts/visualize.sh --image_path ./sample.jpg"
            echo ""
            echo "  # With custom checkpoint"
            echo "  ./scripts/visualize.sh --image_path ./sample.jpg \\"
            echo "      --checkpoint_path ./checkpoint/my_model.pth \\"
            echo "      --model_name MedViT_small"
            echo ""
            echo "  # Save without displaying"
            echo "  ./scripts/visualize.sh --image_path ./sample.jpg \\"
            echo "      --output_path ./result.png --no-show"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if image path is provided
if [ -z "$IMAGE_PATH" ]; then
    echo "Error: --image_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image not found at $IMAGE_PATH"
    exit 1
fi

# Print configuration
echo "Model: $MODEL_NAME"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Classes: $NUM_CLASSES"
echo "Image: $IMAGE_PATH"
echo "Output: $OUTPUT_PATH"
echo "Colormap: $CMAP"
echo "Alpha: $ALPHA"
echo "================================================"
echo ""

# Run visualization
uv run python src/visualize_attention.py \
    --image_path "$IMAGE_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model_name "$MODEL_NAME" \
    --num_classes "$NUM_CLASSES" \
    --output_path "$OUTPUT_PATH" \
    --cmap "$CMAP" \
    --alpha "$ALPHA" \
    $NO_SHOW

echo ""
echo "================================================"
echo "Visualization completed!"
if [ -f "$OUTPUT_PATH" ]; then
    echo "Saved to: $OUTPUT_PATH"
fi
echo "================================================"

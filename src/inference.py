import os
import sys
import warnings
import argparse
import json
import timm
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset, Dataset, DatasetDict
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

from models.MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from dataset_builder import GenericImageDataset, infer_num_classes
from utils import str2bool

# Suppress flex_attention warnings
warnings.filterwarnings("ignore", message=".*return_lse is deprecated.*")
warnings.filterwarnings(
    "ignore", message=".*flex_attention called without torch.compile.*"
)


model_classes = {
    "MedViT_tiny": MedViT_tiny,
    "MedViT_small": MedViT_small,
    "MedViT_base": MedViT_base,
    "MedViT_large": MedViT_large,
}


def get_device():
    """
    Automatically detect and return the best available device.
    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The best available device for inference
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
        return device

    # Try MPS (Apple Silicon) with fallback to CPU
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon) device")
            return device
    except Exception as e:
        print(f"MPS device detection failed: {e}")
        print("Falling back to CPU")

    # Default to CPU
    device = torch.device("cpu")
    print("Using CPU device")
    return device


def load_model(model_name, num_classes, weights_path, device):
    """
    Load a model with pretrained weights.

    Args:
        model_name: Name of the model (MedViT_* or timm model name)
        num_classes: Number of output classes
        weights_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model: {model_name}")
    print(f"Number of classes: {num_classes}")
    print(f"Weights path: {weights_path}")

    # Select model
    if model_name in model_classes:
        model_class = model_classes[model_name]
        net = model_class(num_classes=num_classes).to(device)
    else:
        net = timm.create_model(
            model_name, pretrained=False, num_classes=num_classes
        ).to(device)

    # Load checkpoint
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

    print(f"Loading checkpoint from: {weights_path}")
    # Note: weights_only=False is used because we trust our own trained checkpoints
    # which may contain optimizer state and other objects. Only load checkpoints from trusted sources.
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Extract model state dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        epoch = checkpoint.get("epoch", "unknown")
        acc = checkpoint.get("acc", "unknown")
        print(f"Checkpoint info - Epoch: {epoch}, Accuracy: {acc}")
    else:
        state_dict = checkpoint

    # Load state dict
    net.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully")

    # Set to evaluation mode
    net.eval()
    return net


def run_inference(model, dataloader, device, label_names=None):
    """
    Run inference on a dataset.

    Args:
        model: The model to use for inference
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        label_names: List of label names (optional)

    Returns:
        Dictionary containing predictions, probabilities, labels, and class names
    """
    all_preds = []
    all_labels = []
    all_probs = []

    print("\nRunning inference...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference", file=sys.stdout):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    results = {
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
    }

    if label_names:
        results["predicted_class_names"] = [label_names[pred] for pred in all_preds]
        results["true_class_names"] = [label_names[label] for label in all_labels]
        results["class_names"] = label_names

    return results


def specificity_per_class(conf_matrix):
    """
    Calculates specificity for each class.

    Args:
        conf_matrix: Confusion matrix

    Returns:
        List of specificity values for each class
    """
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (
            conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i]
        )
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        if (tn + fp) > 0:
            specificity.append(tn / (tn + fp))
        else:
            specificity.append(0.0)
    return specificity


def compute_metrics(results, num_classes, comprehensive=False):
    """
    Compute metrics from inference results.

    Args:
        results: Dictionary containing predictions, probabilities, and labels
        num_classes: Number of classes
        comprehensive: If True, compute comprehensive metrics (precision, recall, F1, etc.)

    Returns:
        Dictionary of metrics
    """
    preds = np.array(results["predictions"])
    labels = np.array(results["labels"])
    probs = np.array(results["probabilities"])

    metrics = {}

    if not comprehensive:
        # Basic accuracy only
        accuracy = (preds == labels).mean()
        metrics = {"accuracy": accuracy}

        # Per-class accuracy
        per_class_acc = {}
        for i in range(num_classes):
            mask = labels == i
            if mask.sum() > 0:
                class_acc = (preds[mask] == labels[mask]).mean()
                per_class_acc[f"class_{i}_accuracy"] = class_acc

        metrics.update(per_class_acc)
        return metrics

    # Comprehensive metrics
    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["precision_weighted"] = precision_score(
        labels, preds, average="weighted", zero_division=0
    )
    metrics["recall_weighted"] = recall_score(
        labels, preds, average="weighted", zero_division=0
    )
    metrics["f1_weighted"] = f1_score(
        labels, preds, average="weighted", zero_division=0
    )

    # Per-class metrics
    metrics["precision_per_class"] = precision_score(
        labels, preds, average=None, zero_division=0
    ).tolist()
    metrics["recall_per_class"] = recall_score(
        labels, preds, average=None, zero_division=0
    ).tolist()
    metrics["f1_per_class"] = f1_score(
        labels, preds, average=None, zero_division=0
    ).tolist()

    # Confusion Matrix
    conf_matrix = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = conf_matrix.tolist()

    # Specificity
    specificity = specificity_per_class(conf_matrix)
    metrics["specificity_per_class"] = specificity
    metrics["specificity_weighted"] = np.mean(specificity)

    # Overall accuracy from confusion matrix
    metrics["overall_accuracy"] = conf_matrix.trace() / conf_matrix.sum()

    # AUC-ROC (multi-class)
    try:
        labels_one_hot = label_binarize(labels, classes=list(range(num_classes)))
        if num_classes == 2:
            # Binary classification
            metrics["auc_roc"] = roc_auc_score(labels, probs[:, 1])
        else:
            # Multi-class classification
            metrics["auc_roc"] = roc_auc_score(
                labels_one_hot, probs, multi_class="ovr", average="weighted"
            )
            metrics["auc_roc_per_class"] = roc_auc_score(
                labels_one_hot, probs, multi_class="ovr", average=None
            ).tolist()
    except ValueError as e:
        print(f"Warning: Could not compute AUC-ROC: {e}")
        metrics["auc_roc"] = None
        metrics["auc_roc_per_class"] = None

    # Support (number of samples per class)
    unique, counts = np.unique(labels, return_counts=True)
    metrics["support_per_class"] = dict(zip(unique.tolist(), counts.tolist()))

    return metrics


def print_metrics(metrics, class_names=None, comprehensive=False):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        class_names: List of class names (optional)
        comprehensive: If True, print comprehensive metrics table
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS" if comprehensive else "INFERENCE RESULTS")
    print("=" * 80)

    if not comprehensive:
        # Simple output
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print("\nPer-class Accuracy:")
        for key, value in metrics.items():
            if key.startswith("class_"):
                class_idx = key.split("_")[1]
                class_name = class_names[int(class_idx)] if class_names else class_idx
                print(f"  {class_name}: {value:.4f}")
        print("\n" + "=" * 80)
        return

    # Comprehensive output
    print("\n--- Overall Metrics ---")
    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(f"Overall Accuracy:      {metrics['overall_accuracy']:.4f}")
    print(f"Precision (weighted):  {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):     {metrics['recall_weighted']:.4f}")
    print(f"Specificity (weighted): {metrics['specificity_weighted']:.4f}")
    print(f"F1-Score (weighted):   {metrics['f1_weighted']:.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"AUC-ROC (weighted):    {metrics['auc_roc']:.4f}")

    # Per-class metrics
    print("\n--- Per-Class Metrics ---")
    num_classes = len(metrics["precision_per_class"])

    # Header
    header = f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1-Score':<12} {'Support':<10}"
    if metrics.get("auc_roc_per_class") is not None:
        header += f" {'AUC-ROC':<10}"
    print(header)
    print("-" * len(header))

    # Per-class rows
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        precision = metrics["precision_per_class"][i]
        recall = metrics["recall_per_class"][i]
        specificity = metrics["specificity_per_class"][i]
        f1 = metrics["f1_per_class"][i]
        support = metrics["support_per_class"].get(i, 0)

        row = f"{class_name:<20} {precision:<12.4f} {recall:<12.4f} {specificity:<12.4f} {f1:<12.4f} {support:<10}"
        if metrics.get("auc_roc_per_class") is not None:
            auc = metrics["auc_roc_per_class"][i]
            row += f" {auc:<10.4f}"
        print(row)

    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    conf_matrix = np.array(metrics["confusion_matrix"])

    # Print header
    if class_names:
        header = "True \\ Pred".ljust(15)
        for name in class_names:
            header += f"{name[:10]:>12}"
        print(header)
    else:
        header = "True \\ Pred".ljust(15)
        for i in range(num_classes):
            header += f"Class {i:>12}"
        print(header)

    # Print matrix rows
    for i in range(num_classes):
        if class_names:
            row = f"{class_names[i][:14]:<15}"
        else:
            row = f"Class {i:<9}"

        for j in range(num_classes):
            row += f"{conf_matrix[i, j]:>12}"
        print(row)

    print("\n" + "=" * 80)


def save_metrics(metrics, save_path, class_names=None):
    """
    Save metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the JSON file
        class_names: List of class names (optional)
    """
    output = {
        "metrics": metrics,
    }

    if class_names:
        output["class_names"] = class_names

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nMetrics saved to: {save_path}")


def create_results_dataset(results, original_dataset, image_column="image"):
    """
    Create a HuggingFace dataset with inference results.

    Args:
        results: Dictionary containing predictions and probabilities
        original_dataset: Original HuggingFace dataset split
        image_column: Name of the image column

    Returns:
        HuggingFace Dataset with predictions added
    """
    # Get the original data
    data_dict = {
        "prediction": results["predictions"],
        "probabilities": results["probabilities"],
        "true_label": results["labels"],
    }

    # Add predicted and true class names if available
    if "predicted_class_names" in results:
        data_dict["predicted_class"] = results["predicted_class_names"]
        data_dict["true_class"] = results["true_class_names"]

    # Add images if they exist in the original dataset
    if image_column in original_dataset.column_names:
        data_dict[image_column] = original_dataset[image_column]

    # Create dataset
    dataset = Dataset.from_dict(data_dict)
    return dataset


def main(args):
    device = get_device()

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = hf_load_dataset(args.dataset)

    # Check if split exists
    if args.split not in dataset:
        available_splits = list(dataset.keys())
        raise ValueError(
            f"Split '{args.split}' not found. Available splits: {available_splits}"
        )

    split_dataset = dataset[args.split]
    print(f"Using split: '{args.split}' with {len(split_dataset)} samples")

    # Infer number of classes
    num_classes = infer_num_classes(dataset, label_column=args.label_column)
    print(f"Number of classes: {num_classes}")

    # Get class names if available
    label_names = None
    if hasattr(split_dataset.features[args.label_column], "names"):
        label_names = split_dataset.features[args.label_column].names
        print(f"Class names: {label_names}")

    # Load model
    model = load_model(args.model_name, num_classes, args.model_weights, device)

    # Create dataset and dataloader
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    inference_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    inference_dataset = GenericImageDataset(
        split_dataset,
        transform=inference_transform,
        image_column=args.image_column,
        label_column=args.label_column,
    )

    dataloader = data.DataLoader(
        dataset=inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Run inference
    results = run_inference(model, dataloader, device, label_names)

    # Compute and print metrics
    metrics = compute_metrics(results, num_classes, comprehensive=args.evaluate)
    print_metrics(metrics, label_names, comprehensive=args.evaluate)

    # Save metrics if evaluation mode is enabled
    if args.evaluate:
        os.makedirs("results", exist_ok=True)
        safe_dataset_name = args.dataset.replace("/", "_")
        metrics_path = f"results/{args.model_name}_{safe_dataset_name}_{args.split}_metrics.json"
        save_metrics(metrics, metrics_path, label_names)

    # Save results locally if requested
    if args.save_results:
        os.makedirs("results", exist_ok=True)
        results_dataset = create_results_dataset(
            results, split_dataset, image_column=args.image_column
        )

        # Create a safe filename
        safe_dataset_name = args.dataset.replace("/", "_")
        save_path = f"results/{args.model_name}_{safe_dataset_name}_{args.split}.json"

        # Save as JSON
        results_dataset.to_json(save_path)
        print(f"\nResults saved to: {save_path}")

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        print(f"\nPushing results to HuggingFace Hub: {args.push_to_hub}")
        results_dataset = create_results_dataset(
            results, split_dataset, image_column=args.image_column
        )

        # Create a dataset dict with the results
        results_dataset_dict = DatasetDict({args.split: results_dataset})

        # Push to hub
        try:
            results_dataset_dict.push_to_hub(
                args.push_to_hub,
                private=args.private_hub,
            )
            print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            print("Make sure you are logged in with `huggingface-cli login`")

    print("\nInference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on HuggingFace datasets with trained models."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., MedViT_tiny, MedViT_small, or any timm model).",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to the model weights/checkpoint file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name or path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to run inference on (e.g., test, validation).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Name of the image column in the dataset.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in the dataset.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hub repository name to push results to (e.g., username/dataset-name).",
    )
    parser.add_argument(
        "--private_hub",
        type=str2bool,
        default=False,
        help="Whether to make the Hub repository private.",
    )
    parser.add_argument(
        "--save_results",
        type=str2bool,
        default=True,
        help="Whether to save results locally to results/ directory.",
    )
    parser.add_argument(
        "--evaluate",
        type=str2bool,
        default=False,
        help="Whether to compute comprehensive evaluation metrics (Precision, Recall, F1, AUC-ROC, etc.).",
    )

    args = parser.parse_args()
    main(args)

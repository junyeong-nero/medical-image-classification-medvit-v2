import os
import sys
import warnings
import argparse
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from dataset_builder import build_dataset
from tqdm import tqdm
from medmnist import INFO, Evaluator
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from models.MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from utils import str2bool, model_urls, download_checkpoint

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
        torch.device: The best available device for training
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


# Define the MNIST training routine
def train_mnist(
    epochs,
    net,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    loss_function,
    device,
    save_path,
    data_flag,
    task,
):
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, datax in enumerate(train_bar):
            images, labels = datax
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)

            if task == "multi-label, binary-class":
                labels = labels.to(torch.float32)
                loss = loss_function(outputs, labels)
            else:
                labels = labels.squeeze().long()
                loss = loss_function(outputs.squeeze(0), labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        net.eval()
        y_score = torch.tensor([])
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                inputs, targets = val_data
                outputs = net(inputs.to(device))

                if task == "multi-label, binary-class":
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)

                y_score = torch.cat((y_score, outputs.cpu()), 0)

        y_score = y_score.detach().numpy()
        evaluator = Evaluator(data_flag, "test", size=224, root="./data")
        metrics = evaluator.evaluate(y_score)

        val_accurate, _ = metrics
        print(
            f"[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}"
        )
        # print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        if val_accurate > best_acc:
            best_acc = val_accurate
            print(f"\n✓ New best AUC: {val_accurate:.4f}")
            print(f"  Saving checkpoint to {save_path}")

            # Save full checkpoint (for resuming training)
            state = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "acc": best_acc,
                "epoch": epoch,
            }
            torch.save(state, save_path)

            # Save best model weights only (for inference)
            best_weights_path = save_path.replace(".pth", "_best_weights.pth")
            torch.save(net.state_dict(), best_weights_path)
            print(f"  Best model weights saved to {best_weights_path}")

    print("Finished Training")


# Define the non-MNIST training routine
def specificity_per_class(conf_matrix):
    """Calculates specificity for each class."""
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (
            conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i]
        )
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))
    return specificity


def overall_accuracy(conf_matrix):
    """Calculates overall accuracy for multi-class."""
    tp_tn_sum = conf_matrix.trace()  # Sum of all diagonal elements (TP for all classes)
    total_sum = conf_matrix.sum()  # Sum of all elements in the matrix
    return tp_tn_sum / total_sum


def train(
    epochs,
    net,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    loss_function,
    device,
    save_path,
):
    best_acc = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        # Training Loop
        for step, datax in enumerate(train_bar):
            images, labels = datax
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        # Validation Loop
        net.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # Store raw probabilities/logits for AUC
        acc = 0.0

        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # Raw outputs (logits)
                probs = torch.softmax(outputs, dim=1)  # Convert to probabilities

                predict_y = torch.max(probs, dim=1)[1]  # Predicted class

                # Collect predictions, labels, and probabilities
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Calculate accuracy
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        # Calculate metrics
        val_accurate = acc / len(test_loader.dataset)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")  # Sensitivity
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Confusion Matrix for multi-class
        conf_matrix = confusion_matrix(all_labels, all_preds)
        specificity = specificity_per_class(
            conf_matrix
        )  # List of specificities per class
        avg_specificity = sum(specificity) / len(specificity)  # Average specificity

        # Overall Accuracy calculation
        overall_acc = overall_accuracy(conf_matrix)

        # One-hot encode the labels for AUC calculation
        n_classes = len(conf_matrix)
        all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))

        try:
            # Compute AUC for multi-class
            auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class="ovr")
        except ValueError:
            auc = float("nan")  # Handle edge case where AUC can't be computed

        # Print metrics
        print(
            f"[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f} "
            f"val_accuracy: {val_accurate:.4f} precision: {precision:.4f} "
            f"recall: {recall:.4f} specificity: {avg_specificity:.4f} "
            f"f1_score: {f1:.4f} auc: {auc:.4f} overall_accuracy: {overall_acc:.4f}"
        )

        # print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')

        # Save best model
        if val_accurate > best_acc:
            best_acc = val_accurate
            print(f"\n✓ New best validation accuracy: {best_acc:.4f}")
            print(f"  Saving checkpoint to {save_path}")

            # Save full checkpoint (for resuming training)
            state = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "acc": best_acc,
                "epoch": epoch,
            }
            torch.save(state, save_path)

            # Save best model weights only (for inference)
            best_weights_path = save_path.replace(".pth", "_best_weights.pth")
            torch.save(net.state_dict(), best_weights_path)
            print(f"  Best model weights saved to {best_weights_path}")

    print("Finished Training")


def main(args):
    device = get_device()
    model_name = args.model_name
    dataset_name = args.dataset
    pretrained = args.pretrained

    # Create directories for weights
    os.makedirs("weights", exist_ok=True)
    os.makedirs("weights/pretrained", exist_ok=True)
    if args.dataset.endswith("mnist"):
        info = INFO[args.dataset]
        task = info["task"]
        if task == "multi-label, binary-class":
            loss_function = nn.BCEWithLogitsLoss()
        else:
            loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    batch_size = args.batch_size
    lr = args.lr

    train_dataset, test_dataset, nb_classes = build_dataset(args=args)
    train_num = len(train_dataset)

    # scheduler max iteration
    eta = args.epochs * train_num // args.batch_size

    # Select model
    if model_name in model_classes:
        model_class = model_classes[model_name]
        net = model_class(num_classes=nb_classes).to(device)
        if pretrained:
            checkpoint_path = args.checkpoint_path
            if not os.path.exists(checkpoint_path):
                checkpoint_url = model_urls.get(model_name)
                if not checkpoint_url:
                    raise ValueError(
                        f"Checkpoint URL for model {model_name} not found."
                    )
                # Download to weights/pretrained/
                pretrained_path = f"weights/pretrained/{model_name}.pth"
                download_checkpoint(checkpoint_url, pretrained_path)
                checkpoint_path = pretrained_path

            print(f"Loading checkpoint from: {checkpoint_path}")
            # Note: weights_only=False is used because we trust our own pretrained checkpoints
            # which may contain numpy objects. Only load checkpoints from trusted sources.
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            state_dict = net.state_dict()
            for k in ["proj_head.0.weight", "proj_head.0.bias"]:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            net.load_state_dict(checkpoint, strict=False)
            print("Checkpoint loaded successfully")
    else:
        net = timm.create_model(
            model_name, pretrained=pretrained, num_classes=nb_classes
        ).to(device)

    optimizer = optim.AdamW(
        net.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eta, eta_min=5e-6)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Reuse workers across epochs
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # print(train_dataset)
    # print("===================")
    # print(test_dataset)

    epochs = args.epochs
    # Save trained models to weights/ directory
    safe_dataset_name = dataset_name.replace("/", "_")
    save_path = f"weights/{model_name}_{safe_dataset_name}.pth"

    if dataset_name.endswith("mnist"):
        train_mnist(
            epochs,
            net,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            loss_function,
            device,
            save_path,
            dataset_name,
            task,
        )
    else:
        train(
            epochs,
            net,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            loss_function,
            device,
            save_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for MedViT models.")
    parser.add_argument(
        "--model_name", type=str, default="MedViT_tiny", help="Model name to use."
    )
    # tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, breastmnist, bloodmnist,
    # organamnist, organcmnist, organsmnist'
    parser.add_argument("--dataset", type=str, default="PAD", help="Dataset to use.")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--pretrained",
        type=str2bool,
        default=False,
        help="Whether to use pretrained weights (True/False).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="weights/pretrained/MedViT_tiny.pth",
        help="Path to the pretrained checkpoint file.",
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

    args = parser.parse_args()
    main(args)

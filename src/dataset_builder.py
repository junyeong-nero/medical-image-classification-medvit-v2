import torch
from torchvision import transforms
from datasets import load_dataset as hf_load_dataset


class GenericImageDataset(torch.utils.data.Dataset):
    """Generic dataset wrapper for HuggingFace image datasets."""

    def __init__(
        self, hf_dataset, transform=None, image_column="image", label_column="label"
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset split
            transform: torchvision transforms to apply to images
            image_column: name of the column containing images
            label_column: name of the column containing labels
        """
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

        # Build label-to-index mapping for encoding string labels
        self.label_to_idx = {}
        self.idx_to_label = {}
        self._build_label_mapping()

    def _build_label_mapping(self):
        """Build label-to-index mapping from the dataset."""
        # Get unique labels from the dataset
        unique_labels = set()
        for idx in range(len(self.hf_dataset)):
            label = self.hf_dataset[idx][self.label_column]
            unique_labels.add(label)

        # Create mapping
        for idx, label in enumerate(sorted(unique_labels)):
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item[self.image_column]
        label = item[self.label_column]

        # Convert to RGB if needed
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Encode label: convert string to integer index
        if isinstance(label, str):
            label = self.label_to_idx[label]

        return image, label


def get_dataset_splits(dataset):
    """
    Determine available splits in the dataset.
    Returns: (train_split_name, val_split_name) or raises error if train not found.
    """
    available_splits = list(dataset.keys())

    # Find train split
    train_split = None
    for name in ["train", "training"]:
        if name in available_splits:
            train_split = name
            break

    if train_split is None:
        raise ValueError(
            f"No training split found. Available splits: {available_splits}. "
            "Expected 'train' or 'training'."
        )

    # Find validation/test split
    val_split = None
    for name in ["validation", "val", "valid", "test"]:
        if name in available_splits:
            val_split = name
            break

    if val_split is None:
        raise ValueError(
            f"No validation/test split found. Available splits: {available_splits}. "
            "Expected 'validation', 'val', 'valid', or 'test'."
        )

    return train_split, val_split


def infer_num_classes(dataset, label_column="label"):
    """
    Infer the number of classes from the dataset.

    Args:
        dataset: HuggingFace dataset
        label_column: name of the label column

    Returns:
        Number of classes (int)
    """
    # Get the first split available
    first_split = list(dataset.keys())[0]
    split_data = dataset[first_split]

    # Try to get from features (ClassLabel)
    if hasattr(split_data.features[label_column], "num_classes"):
        return split_data.features[label_column].num_classes

    # Fallback: count unique labels in the split
    labels = split_data[label_column]
    unique_labels = set(labels)
    return len(unique_labels)


def build_dataset(args):
    """
    Build train and test datasets from HuggingFace datasets.

    Args:
        args: Arguments containing:
            - dataset: dataset name or path
            - image_column: name of the image column (default: "image")
            - label_column: name of the label column (default: "label")

    Returns:
        train_dataset, test_dataset, num_classes
    """
    dataset_name = args.dataset
    image_column = getattr(args, "image_column", "image")
    label_column = getattr(args, "label_column", "label")

    print(f"Loading dataset: {dataset_name}")
    print(f"Image column: {image_column}")
    print(f"Label column: {label_column}")

    # Load the dataset
    dataset = hf_load_dataset(dataset_name)

    # Get split names
    train_split, val_split = get_dataset_splits(dataset)
    print(f"Using splits - Train: '{train_split}', Validation: '{val_split}'")

    # Define transforms
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Create dataset wrappers
    train_ds = GenericImageDataset(
        dataset[train_split],
        transform=train_transform,
        image_column=image_column,
        label_column=label_column,
    )

    test_ds = GenericImageDataset(
        dataset[val_split],
        transform=val_transform,
        image_column=image_column,
        label_column=label_column,
    )

    # Get number of classes
    nb_classes = infer_num_classes(dataset, label_column=label_column)
    print(f"Number of classes: {nb_classes}")

    return train_ds, test_ds, nb_classes

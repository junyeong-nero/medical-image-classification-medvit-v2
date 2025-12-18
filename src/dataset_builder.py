import torch
from torchvision import transforms
from datasets import load_dataset as hf_load_dataset

class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.hf_dataset)
        
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image.convert("RGB"))
            
        return image, label

def build_dataset(args):
    dataset_name = args.dataset
    
    if dataset_name == "PranomVignesh/MRI-Images-of-Brain-Tumor":
        print(f"Loading dataset: {dataset_name}")
        # Load the dataset
        dataset = hf_load_dataset(dataset_name)
        
        # Define transforms
        # Standard ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Create dataset wrappers
        train_ds = BrainTumorDataset(dataset['train'], transform=train_transform)
        
        if 'validation' in dataset:
            test_ds = BrainTumorDataset(dataset['validation'], transform=val_transform)
        else:
            # Fallback if no validation split
            test_ds = BrainTumorDataset(dataset['test'], transform=val_transform)
            
        # Get number of classes
        # The dataset features usually contain the ClassLabel info
        nb_classes = dataset['train'].features['label'].num_classes
        
        return train_ds, test_ds, nb_classes

    else:
        # Fallback or error for other datasets if needed
        # For now, we only support the requested one or fail
        raise ValueError(f"Dataset {dataset_name} not supported in build_dataset.")

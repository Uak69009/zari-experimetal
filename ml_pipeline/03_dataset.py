import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

class AddGaussianNoise(object):
    """
    Custom Transform to add Gaussian noise to the image tensor.
    Simulates low-quality smartphone sensors commonly found in the field.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Tensor is in range [0, 1] (or close) after ToTensor()
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class ZariDataset(Dataset):
    """
    Custom PyTorch Dataset for Agricultural Disease Classification.
    Reads from the unified dataset_master.csv.
    """
    def __init__(self, csv_path, split='train', transform=None):
        """
        Args:
            csv_path (string): Path to dataset_master.csv.
            split (string): 'train', 'val', or 'test'.
            transform (callable, optional): Transform pipeline to apply to samples.
        """
        self.split = split
        self.transform = transform
        
        # Load Master Ledger
        df = pd.read_csv(csv_path)
        
        # 1. Build consistent global mapping from the entire dataset
        # Sorting ensures that the mapping (0 to 137) is deterministic and identical across all splits
        unique_labels = sorted(df['unified_label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 2. Filter dataset for the requested split
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        # 3. Extract properties to lists/arrays for fast __getitem__ access
        self.image_paths = self.df['image_path'].values
        self.labels = [self.label_to_idx[lbl] for lbl in self.df['unified_label'].values]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Force RGB conversion to handle grayscale anomalies (e.g. 1-channel images)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Graceful fallback for corrupted images (though our EDA Integrity check should have caught most)
            print(f"Warning: Failed to load {img_path}. Error: {e}")
            image = Image.new('RGB', (256, 256), (0, 0, 0))
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms():
    """
    Returns the train and val/test transformations pipelines.
    """
    # Standard ImageNet stats for normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=0.05), # Smartphone noise simulation
        normalize,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)) # Occlusion/shadow simulation
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_test_transform


def get_dataloaders(csv_path, batch_size=32, num_workers=4):
    """
    Constructs and returns DataLoaders for train, val, and test.
    Handles the extreme class imbalance by applying a WeightedRandomSampler to the train loader.
    """
    print(f"\n--- Initializing Datasets from {csv_path} ---")
    train_transform, val_transform = get_transforms()
    
    # Initialize Datasets
    train_dataset = ZariDataset(csv_path, split='train', transform=train_transform)
    val_dataset = ZariDataset(csv_path, split='val', transform=val_transform)
    test_dataset = ZariDataset(csv_path, split='test', transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    # ---------------------------------------------------------
    # Computing Weights for Extreme Class Imbalance Mitigation
    # ---------------------------------------------------------
    print("\n--- Computing Class Weights for WeightedRandomSampler ---")
    
    # Count frequency of each class in the training split
    class_counts = np.bincount(train_dataset.labels, minlength=len(train_dataset.label_to_idx))
    
    # Safe fallback: if a class has 0 samples, assign count=1 to avoid DivisionByZero
    class_counts = np.where(class_counts == 0, 1, class_counts)
    
    # Weight formula: w = 1.0 / count
    class_weights = 1.0 / class_counts
    
    # Assign the corresponding weight to every single image in the dataset
    sample_weights = np.array([class_weights[label] for label in train_dataset.labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    # Initialize the Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Log top 3 highest weights (Minority) and lowest weights (Majority)
    sorted_classes = np.argsort(class_weights)
    print("\nTop 3 Minority Classes (Heaviest Weights):")
    for idx in sorted_classes[-3:][::-1]:
        print(f"  - {train_dataset.idx_to_label[idx]}: count={class_counts[idx]}, weight={class_weights[idx]:.6f}")
        
    print("\nTop 3 Majority Classes (Lightest Weights):")
    for idx in sorted_classes[:3]:
        print(f"  - {train_dataset.idx_to_label[idx]}: count={class_counts[idx]}, weight={class_weights[idx]:.6f}")
    
    # ---------------------------------------------------------
    # Creating DataLoaders
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,   # IMPORTANT: shuffle MUST be False when using a sampler
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.label_to_idx


if __name__ == "__main__":
    # Windows Multi-processing safety
    import multiprocessing
    multiprocessing.freeze_support()
    
    csv_file = os.path.join(os.path.dirname(__file__), "data", "dataset_master.csv")
    
    if os.path.exists(csv_file):
        # We use num_workers=0 here just to test safely without hanging on Windows
        train_loader, val_loader, test_loader, class_map = get_dataloaders(csv_file, batch_size=16, num_workers=0)
        
        print("\nTesting dataloader retrieval...")
        batch_images, batch_labels = next(iter(train_loader))
        print(f"Batch images shape: {batch_images.shape} (Batch Size, Channels, Height, Width)")
        print(f"Batch labels shape: {batch_labels.shape}")
        
        print(f"Total Unique Classes: {len(class_map)}")
    else:
        print(f"Error: {csv_file} not found.")

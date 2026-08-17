import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
TAXONOMY_PATH = os.path.join(DATA_DIR, "taxonomy.json")

def get_transforms(is_train: bool = True):
    """Returns the transformation pipeline for training or validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class ZariDataset(Dataset):
    def __init__(self, root_dir: str = RAW_DIR, transform=None):
        """
        Initializes the dataset by loading the taxonomy and crawling raw folders.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load Taxonomy JSON
        if not os.path.exists(TAXONOMY_PATH):
            raise FileNotFoundError(f"Taxonomy JSON not found at {TAXONOMY_PATH}")
            
        with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)
            
        self.dataset_mapping = self.taxonomy.get('dataset_mapping', {})
        self.samples = []
        
        self._index_dataset()

    def _index_dataset(self):
        """Crawls raw folders and builds the index of valid images with canonical IDs."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        # Datasets expected in raw directory
        expected_datasets = ['nwrd', 'plantcity', 'plantvillage', 'plantdoc']
        
        for dataset_name in expected_datasets:
            dataset_path = os.path.join(self.root_dir, dataset_name)
            if not os.path.exists(dataset_path):
                continue
                
            dataset_map = self.dataset_mapping.get(dataset_name, {})
            
            # Iterate through categories (raw folder names)
            for category_folder in os.listdir(dataset_path):
                category_path = os.path.join(dataset_path, category_folder)
                
                if not os.path.isdir(category_path) or category_folder not in dataset_map:
                    continue
                    
                canonical_id = dataset_map[category_folder]
                
                # Iterate through images
                for img_name in os.listdir(category_path):
                    ext = os.path.splitext(img_name)[1].lower()
                    if ext in valid_extensions:
                        img_path = os.path.join(category_path, img_name)
                        self.samples.append((img_path, canonical_id))
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, canonical_id = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (UnidentifiedImageError, OSError, Exception):
            # Fallback to black tensor on corruption
            image = torch.zeros((3, 224, 224))
            
        return image, canonical_id

class TransformSubset(Dataset):
    """A dataset wrapper that applies a specific transform to a subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        # We bypass the parent dataset's __getitem__ to apply our own transform
        # by extracting the underlying path directly from the base dataset.
        base_dataset = self.subset.dataset
        original_idx = self.subset.indices[idx]
        img_path, canonical_id = base_dataset.samples[original_idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (UnidentifiedImageError, OSError, Exception):
            image = torch.zeros((3, 224, 224))
            
        return image, canonical_id
        
    def __len__(self):
        return len(self.subset)

def build_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """Builds Train and Validation DataLoaders with an 80/20 split."""
    # 1. Initialize base dataset without transforms (pure indexing)
    base_dataset = ZariDataset(root_dir=RAW_DIR, transform=None)
    
    # 2. Compute split sizes
    dataset_size = len(base_dataset)
    if dataset_size == 0:
        print("Warning: No images found in dataset. DataLoaders will be empty.")
        train_size, val_size = 0, 0
    else:
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
    
    # 3. Perform 80/20 random split with fixed seed 42
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        base_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    # 4. Wrap subsets with respective train/val transforms
    train_dataset = TransformSubset(train_subset, get_transforms(is_train=True))
    val_dataset = TransformSubset(val_subset, get_transforms(is_train=False))
    
    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Testing ZariDataset and DataLoaders...")
    train_loader, val_loader = build_dataloaders(batch_size=4, num_workers=0)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("Dataset module successfully tested.")

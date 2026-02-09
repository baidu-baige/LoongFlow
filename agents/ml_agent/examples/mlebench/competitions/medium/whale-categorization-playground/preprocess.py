import torch
import numpy as np
import pandas as pd
from typing import Tuple, Any
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Task-adaptive type definitions
# X: Feature set representation (can be WhaleDataset or Subset)
# y: Target set representation (pd.Series or torch.Tensor)
X = Any
y = Any

class PreprocessedWhaleDataset(Dataset):
    """
    Custom Dataset wrapper to apply Albumentations transformations on raw image data.
    Directly accesses the RAM-cached uint8 tensors from the upstream WhaleDataset 
    to maximize performance and avoid redundant normalization.
    """
    def __init__(self, original_dataset: Any, labels: torch.Tensor = None, transform: A.Compose = None):
        self.original_dataset = original_dataset
        self.labels = labels
        self.transform = transform
        
        # Handle Subset wrappers created during the splitting stage
        if isinstance(original_dataset, Subset):
            self.base_ds = original_dataset.dataset
            self.indices = original_dataset.indices
        else:
            self.base_ds = original_dataset
            self.indices = None

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Any:
        # Resolve the index in the original RAM-cached tensor
        actual_idx = self.indices[idx] if self.indices is not None else idx
        
        # Fetch the image tensor (C, H, W) in uint8 [0, 255]
        # Convert to numpy (H, W, C) as required by Albumentations
        img = self.base_ds.images[actual_idx].permute(1, 2, 0).numpy()
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            # Fallback manual normalization if no transform is provided
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
        if self.labels is not None:
            return img, self.labels[idx]
        return img

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw whale image data into model-ready augmented datasets.
    
    Strategy:
    1. Label encode Whale IDs.
    2. Define augmentation pipelines for training (diversity) and validation (consistency).
    3. Wrap raw tensors in PreprocessedWhaleDataset for high-throughput batching.
    """
    print(f"Preprocessing starting. Transforming {len(y_train)} train and {len(y_val)} validation samples...")

    # Step 1: Fit and apply LabelEncoder on training IDs
    # Constraints: Fit on X_train only to prevent leakage. Unseen classes in val will propagate errors.
    le = LabelEncoder()
    le.fit(y_train)
    
    # Transform targets to LongTensors for model consumption
    y_train_processed = torch.tensor(le.transform(y_train), dtype=torch.long)
    y_val_processed = torch.tensor(le.transform(y_val), dtype=torch.long)
    
    # Step 2: Define Augmentation Strategies (Technical Specification)
    # Augmentations: HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=15, p=0.5, border_mode=0), # border_mode=0 is constant
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    # Validation/Test: Consistency only (Normalization)
    val_test_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    # Step 3: Wrap data into transformed Datasets
    # This ensures augmentations are applied on-the-fly during training loop execution
    X_train_processed = PreprocessedWhaleDataset(
        original_dataset=X_train,
        labels=y_train_processed,
        transform=train_transform
    )
    
    X_val_processed = PreprocessedWhaleDataset(
        original_dataset=X_val,
        labels=y_val_processed,
        transform=val_test_transform
    )
    
    X_test_processed = PreprocessedWhaleDataset(
        original_dataset=X_test,
        labels=None,
        transform=val_test_transform
    )
    
    # Step 4: Verification
    # Ensure all test samples are accounted for and alignment is preserved
    if len(X_test_processed) != len(X_test):
        raise RuntimeError("Test set completeness check failed: Samples dropped during preprocessing.")
    
    if len(X_train_processed) != len(y_train_processed):
        raise RuntimeError("Alignment mismatch in processed training data.")

    print(f"Preprocessing complete. Number of classes identified: {len(le.classes_)}")
    
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed
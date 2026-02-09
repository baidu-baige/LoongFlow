import pandas as pd
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import Tuple, Any

# Concrete types for this task
X = Dataset    # PyTorch Dataset for lazy loading and on-the-fly augmentation
y = np.ndarray  # Binary label matrix (N, 6)

class PlantDataset(Dataset):
    """
    Custom Dataset for the Plant Pathology 2021 challenge.
    Loads images from paths and applies Albumentations.
    Returns only the image tensor to satisfy the X/y separation in the pipeline.
    """
    def __init__(self, paths: pd.Series, transform: A.Compose = None):
        self.paths = paths.values if hasattr(paths, 'values') else paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        
        # Load image
        image = cv2.imread(path)
        if image is None:
            # Critical error propagation as per requirements
            raise FileNotFoundError(f"Preprocessing Error: Image not found at {path}")
            
        # Convert BGR (cv2 default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformation pipeline
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image

def preprocess(
    X_train: pd.Series,
    y_train: y,
    X_val: pd.Series,
    y_val: y,
    X_test: pd.Series
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw image paths and binary labels into model-ready PyTorch Datasets.
    
    Implements:
    - High-resolution (512x512) processing to capture fine-grained features.
    - Comprehensive augmentation pipeline (RandomResizedCrop, Flips, Rotation, ColorJitter, ShiftScaleRotate).
    - ImageNet normalization.
    """
    print(f"Preprocessing data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Shared normalization parameters
    norm = A.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225), 
        p=1.0
    )

    # Technical Specification: Training Augmentations
    # Note: Using size=(512, 512) for RandomResizedCrop to comply with newer Albumentations versions.
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        norm,
        ToTensorV2(),
    ])

    # Technical Specification: Validation/Test Transformations
    # Resize to 512, CenterCrop, Normalize
    val_test_transform = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.CenterCrop(height=512, width=512, p=1.0),
        norm,
        ToTensorV2(),
    ])

    # Create dataset objects
    # We maintain y separately as requested by the function signature
    X_train_processed = PlantDataset(X_train, transform=train_transform)
    y_train_processed = y_train.astype(np.float32)

    X_val_processed = PlantDataset(X_val, transform=val_test_transform)
    y_val_processed = y_val.astype(np.float32)

    X_test_processed = PlantDataset(X_test, transform=val_test_transform)

    # Validation of output constraints
    assert len(X_train_processed) == len(y_train_processed), "Alignment Error: X_train and y_train size mismatch"
    assert len(X_val_processed) == len(y_val_processed), "Alignment Error: X_val and y_val size mismatch"
    assert len(X_test_processed) == len(X_test), "Completeness Error: Test samples dropped"

    print("Preprocessing complete. Datasets ready for model training.")
    
    return (
        X_train_processed, 
        y_train_processed, 
        X_val_processed, 
        y_val_processed, 
        X_test_processed
    )
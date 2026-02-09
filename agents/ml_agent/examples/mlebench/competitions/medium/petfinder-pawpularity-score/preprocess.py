import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Any, List

# Task-adaptive type definitions
# X is defined as the Dataset object for features/images
# y is defined as the DataFrame for labels
X = Any      
y = Any      

class PetDataset(Dataset):
    """
    PyTorch Dataset for PetFinder Pawpularity.
    Yields (image, metadata_vector, target) for training/validation,
    and (image, metadata_vector) for testing.
    """
    def __init__(self, 
                 image_paths: np.ndarray, 
                 metadata: np.ndarray, 
                 targets: np.ndarray = None, 
                 transform: A.Compose = None):
        self.image_paths = image_paths
        self.metadata = metadata
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at: {img_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations/transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Metadata vector (features)
        meta = torch.tensor(self.metadata[idx], dtype=torch.float32)
        
        # If targets are provided (Train/Val mode)
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, meta, target
        
        # If no targets (Test mode)
        return image, meta

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready format for a single fold/split.
    
    Returns:
        X_train_dataset (PetDataset): Training set as a PyTorch Dataset.
        y_train (pd.DataFrame): Training targets.
        X_val_dataset (PetDataset): Validation set as a PyTorch Dataset.
        y_val (pd.DataFrame): Validation targets.
        X_test_dataset (PetDataset): Test set as a PyTorch Dataset.
    """
    print("Starting preprocessing stage...")
    
    # Identify metadata columns (features) - exclude the image path
    meta_features = [col for col in X_train.columns if col != 'image_path']
    print(f"Detected {len(meta_features)} metadata features: {meta_features}")

    # Define Albumentations pipeline for training
    # Parameters matched to Technical Specification: 384x384, Mixup ready.
    train_transform = A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # Define Albumentations pipeline for validation and testing (no augmentations)
    val_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # Ensure no NaN values in metadata features (though EDA suggests clean data)
    # Using simple zero-fill to maintain pipeline robustness
    X_train_meta = X_train[meta_features].fillna(0).values
    X_val_meta = X_val[meta_features].fillna(0).values
    X_test_meta = X_test[meta_features].fillna(0).values

    # Extract target values (Multi-task: Pawpularity_scaled + metadata labels)
    y_train_values = y_train.values
    y_val_values = y_val.values

    # Construct PyTorch Dataset objects
    print("Creating PyTorch Dataset objects...")
    
    X_train_dataset = PetDataset(
        image_paths=X_train['image_path'].values,
        metadata=X_train_meta,
        targets=y_train_values,
        transform=train_transform
    )

    X_val_dataset = PetDataset(
        image_paths=X_val['image_path'].values,
        metadata=X_val_meta,
        targets=y_val_values,
        transform=val_transform
    )

    X_test_dataset = PetDataset(
        image_paths=X_test['image_path'].values,
        metadata=X_test_meta,
        targets=None,
        transform=val_transform
    )

    # Sanity checks
    assert len(X_train_dataset) == len(y_train), "X_train and y_train size mismatch"
    assert len(X_val_dataset) == len(y_val), "X_val and y_val size mismatch"
    assert len(X_test_dataset) == len(X_test), "X_test size mismatch"
    
    print(f"Preprocessing complete. Train size: {len(X_train_dataset)}, "
          f"Val size: {len(X_val_dataset)}, Test size: {len(X_test_dataset)}")

    return X_train_dataset, y_train, X_val_dataset, y_val, X_test_dataset
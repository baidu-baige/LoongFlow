# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Standard data path constants (as provided)
BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/7dbf3696-f36d-4f87-8e59-bb45d92869c5/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class DogCatDataset(Dataset):
    """
    Custom Dataset class that takes image paths and labels, applying the appropriate 
    transformation pipeline (train vs. val/test).
    """

    def __init__(self, dataframe: pd.DataFrame, labels: pd.Series = None, is_train: bool = False):
        self.paths = dataframe['path'].values
        self.labels = labels.values if labels is not None else None
        self.is_train = is_train

        # Standard input size for EfficientNet-B0 is 224x224
        # ImageNet normalization constants
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        # Define transformation pipelines
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        # Open image and convert to RGB to handle grayscale images
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label

        return image


def create_features(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT, DT, DT, DT]:
    """
    Standardizes image feature metadata and prepares labels for model training.
    
    The actual image transformation (Resize, Augmentation, Normalization) is 
    encapsulated in the DogCatDataset class which is designed to be used with the 
    returned paths.
    """
    # Step 1: Data Validation - Ensure no missing paths or labels
    # We use copies to prevent side effects on the original dataframes
    X_train_transformed = X_train.copy()
    y_train_transformed = y_train.copy()
    X_val_transformed = X_val.copy()
    y_val_transformed = y_val.copy()
    X_test_transformed = X_test.copy()

    # Step 2: Ensure Column Consistency
    # All transformed feature sets (X_train, X_val, X_test) must have identical feature columns.
    # We maintain the 'path' column as the primary feature identifier.
    cols_to_keep = ['path']
    X_train_transformed = X_train_transformed[cols_to_keep]
    X_val_transformed = X_val_transformed[cols_to_keep]
    X_test_transformed = X_test_transformed[cols_to_keep]

    # Step 3: Standardize labels
    # Ensure labels are float32 for compatibility with PyTorch binary cross-entropy (log loss)
    y_train_transformed = y_train_transformed.astype(float)
    y_val_transformed = y_val_transformed.astype(float)

    # Step 4: Validate output format (no NaN/Inf)
    # Check features (paths)
    if X_train_transformed.isna().any().any() or X_val_transformed.isna().any().any() or X_test_transformed.isna().any().any():
        raise ValueError("Feature paths contain NaN values.")

    # Check labels
    if y_train_transformed.isna().any() or y_val_transformed.isna().any():
        raise ValueError("Labels contain NaN values.")

    # Standardized check for infinite values (non-numeric columns skip this)
    for df in [y_train_transformed, y_val_transformed]:
        if np.isinf(df).any():
            raise ValueError("Labels contain Infinity values.")

    # Step 5: Return the transformed datasets
    # Row preservation is maintained by using the original dataframes and indices.
    return (
        X_train_transformed,
        y_train_transformed,
        X_val_transformed,
        y_val_transformed,
        X_test_transformed
    )

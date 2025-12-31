# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/1799683e-18f2-43a3-97d1-8b0bdddc3200/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class PlantDataset(Dataset):
    """
    Custom Dataset class for loading apple leaf images and applying transformations.
    """

    def __init__(self, df: pd.DataFrame, labels: pd.DataFrame = None, transform: A.Compose = None):
        """
        Args:
            df: DataFrame containing the 'image_path' column.
            labels: DataFrame containing the target labels (one-hot encoded).
            transform: Albumentations transformation pipeline.
        """
        self.image_paths = df['image_path'].values
        self.labels = labels.values if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image using the path derived from image_id in load_data
        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Return image and label if available (training/validation), otherwise just image (testing)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label

        return image


def get_train_transforms() -> A.Compose:
    """
    Defines the augmentation pipeline for training data.
    """
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_valid_transforms() -> A.Compose:
    """
    Defines the augmentation pipeline for validation and test data.
    """
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_features(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT, DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation.
    
    This implementation prepares the image paths and labels for use with the 
    defined PlantDataset class and albumentations pipelines.
    """

    # Step 1: Validate inputs and ensure they are model-ready
    # For image data in this pipeline, the 'features' are the image paths.
    # We ensure that the DataFrames are consistent and contain no missing values.

    for df, name in zip([X_train, X_val, X_test], ["X_train", "X_val", "X_test"]):
        if df.isnull().any().any():
            raise ValueError(f"{name} contains NaN values in paths.")

    for df, name in zip([y_train, y_val], ["y_train", "y_val"]):
        if df.isnull().any().any():
            raise ValueError(f"{name} contains NaN values in labels.")

    # Step 2: Transformation implementation
    # The actual 'transformation' (loading, resizing, augmenting, normalizing) is
    # encapsulated in the PlantDataset class and the helper functions 
    # get_train_transforms() and get_valid_transforms(). 
    # These are intended to be used by the downstream training component.

    # We return the DataFrames as-is, ensuring they maintain row preservation 
    # and column consistency as required by the pipeline.

    X_train_transformed = X_train.copy()
    y_train_transformed = y_train.copy()
    X_val_transformed = X_val.copy()
    y_val_transformed = y_val.copy()
    X_test_transformed = X_test.copy()

    # Step 3: Final validation of output format
    # Check column consistency across feature sets
    assert list(X_train_transformed.columns) == list(X_val_transformed.columns) == list(X_test_transformed.columns)

    # Check row preservation
    assert len(X_train_transformed) == len(X_train)
    assert len(y_train_transformed) == len(y_train)
    assert len(X_val_transformed) == len(X_val)
    assert len(y_val_transformed) == len(y_val)
    assert len(X_test_transformed) == len(X_test)

    return (
        X_train_transformed,
        y_train_transformed,
        X_val_transformed,
        y_val_transformed,
        X_test_transformed
    )

# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/ee5d67f1-b3aa-447c-b45e-155ce5c4f09c/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class HistopathDataset(Dataset):
    """
    PyTorch Dataset for loading histopathology images.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        # Load image using cv2 for speed
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL for torchvision transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Return dummy label if not present (for test set)
        label = self.df.loc[idx, 'label'] if 'label' in self.df.columns else 0
        return image, label


def get_transforms(mode: str = 'train'):
    """
    Defines the transformation pipeline for training and validation/test.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # 90-degree rotations to avoid interpolation artifacts
            transforms.RandomChoice([
                transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: TF.rotate(x, 90)),
                transforms.Lambda(lambda x: TF.rotate(x, 180)),
                transforms.Lambda(lambda x: TF.rotate(x, 270))
            ]),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])


def _compute_single_img_stats(img_path: str) -> np.ndarray:
    """
    Helper function to extract image statistics for feature engineering.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Global stats
    mean = img.mean(axis=(0, 1)) / 255.0
    std = img.std(axis=(0, 1)) / 255.0

    # Center 32x32 stats (tumor detection area)
    # Image is 96x96, center starts at (96-32)//2 = 32
    center = img[32:64, 32:64]
    c_mean = center.mean(axis=(0, 1)) / 255.0
    c_std = center.std(axis=(0, 1)) / 255.0

    return np.concatenate([mean, std, c_mean, c_std])


def extract_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts color statistics as tabular features for the model.
    """
    paths = df['path'].tolist()
    # Using ProcessPoolExecutor for parallel processing across many cores
    # Limit workers to 32 to balance overhead and throughput
    num_workers = min(32, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        stats_list = list(executor.map(_compute_single_img_stats, paths))

    stats_cols = [
        'mean_r', 'mean_g', 'mean_b',
        'std_r', 'std_g', 'std_b',
        'center_mean_r', 'center_mean_g', 'center_mean_b',
        'center_std_r', 'center_std_g', 'center_std_b'
    ]

    stats_df = pd.DataFrame(stats_list, columns=stats_cols, index=df.index)

    # Ensure no NaNs or Infs
    if stats_df.isnull().values.any() or np.isinf(stats_df.values).any():
        stats_df = stats_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return pd.concat([df, stats_df], axis=1)


def create_features(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT, DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation.
    Prepares DataFrames with image metadata and statistical features.
    """
    # 1. Feature Extraction: Calculate image-level statistics as tabular features
    # This provides the model with immediate color/texture information
    X_train_transformed = extract_tabular_features(X_train)
    X_val_transformed = extract_tabular_features(X_val)
    X_test_transformed = extract_tabular_features(X_test)

    # 2. Label Preservation
    # Labels remain unchanged but ensure they are consistent in type
    y_train_transformed = y_train.copy()
    y_val_transformed = y_val.copy()

    # 3. Validation: Consistency checks
    # Row counts must be preserved
    assert len(X_train_transformed) == len(X_train)
    assert len(X_val_transformed) == len(X_val)
    assert len(X_test_transformed) == len(X_test)

    # Column consistency
    assert all(X_train_transformed.columns == X_val_transformed.columns)
    assert all(X_train_transformed.columns == X_test_transformed.columns)

    # No NaN/Inf values in the new features
    for df in [X_train_transformed, X_val_transformed, X_test_transformed]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if df[numeric_cols].isnull().values.any() or np.isinf(df[numeric_cols].values).any():
            raise ValueError("Feature extraction produced NaN or Infinity values.")

    return (
        X_train_transformed,
        y_train_transformed,
        X_val_transformed,
        y_val_transformed,
        X_test_transformed
    )

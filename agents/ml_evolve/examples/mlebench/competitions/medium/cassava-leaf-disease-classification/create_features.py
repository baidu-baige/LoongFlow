# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/068f0c14-e630-462e-bc46-9d2d4b1d5fc3/5/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def create_features(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT, DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation.
    Defines and persists synchronized heavy augmentation pipelines for model training.
    
    Args:
        X_train (DT): The training set features (image IDs).
        y_train (DT): The training set labels.
        X_val (DT): The validation set features (image IDs).
        y_val (DT): The validation set labels.
        X_test (DT): The test set features (image IDs).
    
    Returns:
        Tuple[DT, DT, DT, DT, DT]: Transformed datasets.
    """

    # Step 1: Define heavy training augmentation pipeline.
    # Note: size=(512, 512) is required by newer Albumentations versions for RandomResizedCrop.
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(512, 512), scale=(0.08, 1.0), ratio=(0.75, 1.333), p=1.0),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ToTensorV2(),
    ])

    # Step 2: Define validation/inference augmentation pipeline.
    val_test_transform = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ToTensorV2(),
    ])

    # Step 3: Persist augmentation definitions to disk.
    # This Stage provides the definitions that train_and_predict will use.
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    A.save(train_transform, os.path.join(OUTPUT_DATA_PATH, "train_transform.json"))
    A.save(val_test_transform, os.path.join(OUTPUT_DATA_PATH, "val_test_transform.json"))

    # Step 4: Prepare output metadata DataFrames.
    # The actual image transformations are applied in the Dataset class during the training loop.
    X_train_transformed = X_train.copy()
    y_train_transformed = y_train.copy()
    X_val_transformed = X_val.copy()
    y_val_transformed = y_val.copy()
    X_test_transformed = X_test.copy()

    # Step 5: Validate output format (no NaN/Inf).
    if y_train_transformed.isna().any() or y_val_transformed.isna().any():
        raise ValueError("Target labels contain NaN values.")

    # Cast to float for infinity check
    if np.isinf(y_train_transformed.values.astype(float)).any() or \
            np.isinf(y_val_transformed.values.astype(float)).any():
        raise ValueError("Target labels contain Infinity values.")

    return (
        X_train_transformed,
        y_train_transformed,
        X_val_transformed,
        y_val_transformed,
        X_test_transformed
    )

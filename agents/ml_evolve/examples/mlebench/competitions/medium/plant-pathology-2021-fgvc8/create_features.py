# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/1ae93cb9-976c-4242-a6ab-6fcbc92f5502/1/executor/output"

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
    Creates features for a single fold of cross-validation by transforming images into augmented tensors.
    """

    # Define augmentation pipelines
    # Resize to 380x380 for EfficientNet-B4
    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    train_transform = A.Compose([
        A.Resize(380, 380),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_and_transform(img_name: str, transform: A.Compose, is_test: bool) -> np.ndarray:
        """Helper to load, color-convert, and apply transformations to a single image."""
        img_dir = "test_images" if is_test else "train_images"
        full_path = os.path.join(BASE_DATA_PATH, img_dir, img_name)

        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image {img_name} could not be loaded from {full_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image)
        return transformed['image']

    def batch_process(df: pd.DataFrame, transform: A.Compose, is_test: bool) -> pd.DataFrame:
        """Processes a batch of images in parallel and returns a DataFrame with image features."""
        paths = df['image_path'].tolist()

        # Using ThreadPoolExecutor for efficient I/O and CPU-bound image processing
        # 32 workers is a reasonable balance for 112 cores and disk I/O
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(lambda p: load_and_transform(p, transform, is_test), paths))

        # Store the resulting numpy arrays (tensors) in a single column
        # Ensure the index matches the original input for row preservation
        return pd.DataFrame({'image_features': results}, index=df.index)

    # Step 1 & 2: Apply transformations to train, val, and test sets
    # Training set receives augmentations; Validation and Test sets receive static resizing/normalization
    X_train_transformed = batch_process(X_train, train_transform, is_test=False)
    X_val_transformed = batch_process(X_val, val_test_transform, is_test=False)
    X_test_transformed = batch_process(X_test, val_test_transform, is_test=True)

    # Step 3: Validate output format
    # Labels remain unchanged as they are already binarized in load_data
    y_train_transformed = y_train.copy()
    y_val_transformed = y_val.copy()

    # Final Verification
    # Row preservation check
    if len(X_train_transformed) != len(X_train) or len(X_val_transformed) != len(X_val) or len(
            X_test_transformed) != len(X_test):
        raise ValueError("Transformed datasets do not match original row counts.")

    # Column consistency check
    if not (X_train_transformed.columns.equals(X_val_transformed.columns) and
            X_val_transformed.columns.equals(X_test_transformed.columns)):
        raise ValueError("Transformed feature sets do not have identical columns.")

    # Step 4: Return precisely 5 transformed components
    return X_train_transformed, y_train_transformed, X_val_transformed, y_val_transformed, X_test_transformed

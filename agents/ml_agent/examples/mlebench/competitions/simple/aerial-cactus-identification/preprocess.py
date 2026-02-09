import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from joblib import Parallel, delayed
import torch
from typing import Tuple

# Task-adaptive type definitions
# X represents the feature matrix as a 4D torch.Tensor (N, C, H, W)
# y represents the target label vector as a 1D torch.Tensor (N,)
X = torch.Tensor
y = torch.Tensor


def _process_image(path: str, transform: A.Compose) -> torch.Tensor:
    """
    Helper function to read an image from disk, convert color space, 
    apply albumentations, and return a torch Tensor.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image at: {path}")

    # Convert from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transformation pipeline
    transformed = transform(image=img)
    return transformed['image']


def preprocess(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    X_test: pd.Series
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw image paths into model-ready tensors for a single fold/split.

    - Upscales 32x32 images to 128x128.
    - Applies aggressive augmentation (flips, rotations, jitter) to training data.
    - Normalizes all sets using ImageNet statistics.
    - Leverages multi-core CPU for fast image decoding and transformation.

    Args:
        X_train (pd.Series): Absolute file paths for training images.
        y_train (pd.Series): Binary labels for training images.
        X_val (pd.Series): Absolute file paths for validation images.
        y_val (pd.Series): Binary labels for validation images.
        X_test (pd.Series): Absolute file paths for test images.

    Returns:
        Tuple[X, y, X, y, X]: (X_train_p, y_train_p, X_val_p, y_val_p, X_test_p)
    """
    print(f"Initializing preprocessing for {len(X_train)} train, {len(X_val)} val, and {len(X_test)} test images...")

    # Define ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Step 1: Define Pipelines
    # Training pipeline: Resize -> Augmentations -> Normalize -> Tensor
    train_transform = A.Compose([
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    # Validation/Inference pipeline: Resize -> Normalize -> Tensor (No Augmentation)
    val_test_transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    # Step 2: Parallel Transformation execution
    # Utilize 36 Cores to maximize throughput during IO-heavy image processing
    print("Transforming training set...")
    X_train_list = Parallel(n_jobs=-1)(
        delayed(_process_image)(path, train_transform) for path in X_train
    )
    X_train_processed = torch.stack(X_train_list)
    y_train_processed = torch.tensor(y_train.values, dtype=torch.float32)

    print("Transforming validation set...")
    X_val_list = Parallel(n_jobs=-1)(
        delayed(_process_image)(path, val_test_transform) for path in X_val
    )
    X_val_processed = torch.stack(X_val_list)
    y_val_processed = torch.tensor(y_val.values, dtype=torch.float32)

    print("Transforming test set...")
    X_test_list = Parallel(n_jobs=-1)(
        delayed(_process_image)(path, val_test_transform) for path in X_test
    )
    X_test_processed = torch.stack(X_test_list)

    # Step 3: Integrity Validation
    # Ensure no NaN/Inf were introduced during normalization/resizing
    if torch.isnan(X_train_processed).any() or torch.isinf(X_train_processed).any():
        raise ValueError("Preprocessing error: NaN or Inf detected in training tensors.")

    # Check row alignment and completeness
    assert len(X_train_processed) == len(y_train_processed), "Train features/targets mismatch."
    assert len(X_val_processed) == len(y_val_processed), "Val features/targets mismatch."
    assert len(X_test_processed) == len(X_test), "Test set completeness check failed."

    print(f"Preprocessing completed successfully. Output shape: {X_train_processed.shape}")

    return (
        X_train_processed,
        y_train_processed,
        X_val_processed,
        y_val_processed,
        X_test_processed
    )
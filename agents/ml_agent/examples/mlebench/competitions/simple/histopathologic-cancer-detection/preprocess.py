import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from joblib import Parallel, delayed
from typing import Tuple, Dict, Any, List

# Task-adaptive type definitions
# X is a pandas DataFrame that contains image metadata and the actual image arrays in a column.
# This ensures that len(X) correctly returns the number of samples, matching len(y).
X = pd.DataFrame
y = pd.Series

def load_image(path: str) -> np.ndarray:
    """
    Loads an image from disk and converts it to RGB format.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image at: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw image metadata into model-ready DataFrames containing 
    in-memory image arrays and associated transformation pipelines.
    
    The implementation satisfies the D4 symmetry group requirement through 
    specific geometric augmentations and 8-fold TTA configurations.
    """
    print("Starting preprocessing stage...")

    # 1. Define Augmentation Pipelines
    # Inductive bias: Histopathology slides are orientation-agnostic (D4 symmetry).
    # Albumentations.Flip is replaced with HorizontalFlip and VerticalFlip for compatibility.
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Validation and default Test transform (Standard normalization, no geometric augmentation)
    val_test_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 8-fold TTA (Test Time Augmentation) covering the D4 symmetry group.
    # The D4 group elements: {Id, R90, R180, R270, FlipH, FlipV, Transpose, Transverse}.
    tta_transforms = [
        # 1. Identity
        A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
        # 2. Rotate 90
        A.Compose([A.Rotate(limit=(90, 90), p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
        # 3. Rotate 180
        A.Compose([A.Rotate(limit=(180, 180), p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
        # 4. Rotate 270
        A.Compose([A.Rotate(limit=(270, 270), p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
        # 5. Horizontal Flip
        A.Compose([A.HorizontalFlip(p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
        # 6. Vertical Flip
        A.Compose([A.VerticalFlip(p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
        # 7. Transpose
        A.Compose([A.Transpose(p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]),
        # 8. Transverse (Rotate 90 + Horizontal Flip)
        A.Compose([A.Rotate(limit=(90, 90), p=1.0), A.HorizontalFlip(p=1.0), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    ]

    # 2. Parallel Image Loading
    # Leveraging 36 cores and high RAM to load images into memory for maximum training throughput.
    def load_all_parallel(paths: pd.Series) -> List[np.ndarray]:
        return Parallel(n_jobs=-1)(delayed(load_image)(p) for p in paths)

    print(f"Loading {len(X_train)} training images into RAM...")
    X_train_imgs = load_all_parallel(X_train['path'])
    
    print(f"Loading {len(X_val)} validation images into RAM...")
    X_val_imgs = load_all_parallel(X_val['path'])
    
    print(f"Loading {len(X_test)} test images into RAM...")
    X_test_imgs = load_all_parallel(X_test['path'])

    # 3. Construct Model-Ready DataFrames
    # We use a DataFrame with an 'image' column to ensure len(X) matches len(y) for the pipeline.
    # We use .attrs to attach the transformation policies for the downstream trainer.
    
    X_train_processed = pd.DataFrame({
        'id': X_train['id'].values,
        'image': X_train_imgs
    }, index=X_train.index)
    X_train_processed.attrs['transform'] = train_transform

    X_val_processed = pd.DataFrame({
        'id': X_val['id'].values,
        'image': X_val_imgs
    }, index=X_val.index)
    X_val_processed.attrs['transform'] = val_test_transform

    X_test_processed = pd.DataFrame({
        'id': X_test['id'].values,
        'image': X_test_imgs
    }, index=X_test.index)
    X_test_processed.attrs['transform'] = val_test_transform
    X_test_processed.attrs['tta_transforms'] = tta_transforms

    # Final verification of row alignment and data integrity
    assert len(X_train_processed) == len(y_train), f"Alignment Error: X_train ({len(X_train_processed)}) != y_train ({len(y_train)})"
    assert len(X_val_processed) == len(y_val), f"Alignment Error: X_val ({len(X_val_processed)}) != y_val ({len(y_val)})"
    assert len(X_test_processed) == len(X_test), "Test completeness error: dropped samples detected."
    
    if y_train.isna().any() or y_val.isna().any():
        raise ValueError("Target labels contain NaN values.")

    print(f"Preprocessing complete. All image sets aligned and loaded into RAM.")
    
    return X_train_processed, y_train, X_val_processed, y_val, X_test_processed
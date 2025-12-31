# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/aerial-cactus-identification/prepared/public"
OUTPUT_DATA_PATH = "output/153e4624-940b-4d19-a37d-90435531bfd1/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def create_features(
    X_train: DT,
    y_train: DT,
    X_test: DT
) -> Tuple[DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation.
    
    Applies ImageNet normalization to the image data.
    Note: Random augmentations (flips, rotations, color jitter) should be applied
    during training time in the DataLoader for proper randomization per epoch.
    
    Args:
        X_train (DT): The training set features (flattened 32x32x3 images).
        y_train (DT): The training set labels.
        X_test (DT): The test/validation set features (flattened 32x32x3 images).

    Returns:
        Tuple[DT, DT, DT]: Transformed training features, labels, and test features.
    """
    # ImageNet normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Image dimensions
    height, width, channels = 32, 32, 3
    num_pixels_per_channel = height * width  # 1024

    def apply_normalization(X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ImageNet normalization to flattened image data.
        Input is in [0, 1] range, flattened as (R, G, B) interleaved or sequential.
        """
        # Convert to numpy for faster processing
        X_np = X.values.astype(np.float32)

        # The data is flattened from shape (32, 32, 3)
        # When flattened with .flatten(), it goes row by row: 
        # pixel[0,0,0], pixel[0,0,1], pixel[0,0,2], pixel[0,1,0], ...
        # This means RGB values are interleaved

        # Reshape to (n_samples, height, width, channels)
        n_samples = X_np.shape[0]
        X_reshaped = X_np.reshape(n_samples, height, width, channels)

        # Apply normalization per channel
        # Normalize: (pixel - mean) / std
        X_normalized = np.zeros_like(X_reshaped)
        for c in range(channels):
            X_normalized[:, :, :, c] = (X_reshaped[:, :, :, c] - mean[c]) / std[c]

        # Flatten back to original shape
        X_flat = X_normalized.reshape(n_samples, -1)

        # Convert back to DataFrame with same column names
        return pd.DataFrame(X_flat, columns=X.columns, index=X.index)

    # Apply normalization to training data
    X_train_transformed = apply_normalization(X_train)

    # Apply normalization to test data
    X_test_transformed = apply_normalization(X_test)

    # Labels remain unchanged
    y_train_transformed = y_train.copy()

    # Verify no NaN or Infinity values
    assert not X_train_transformed.isnull().any().any(), "NaN values in X_train_transformed"
    assert not X_test_transformed.isnull().any().any(), "NaN values in X_test_transformed"
    assert not np.isinf(X_train_transformed.values).any(), "Inf values in X_train_transformed"
    assert not np.isinf(X_test_transformed.values).any(), "Inf values in X_test_transformed"

    return X_train_transformed, y_train_transformed, X_test_transformed

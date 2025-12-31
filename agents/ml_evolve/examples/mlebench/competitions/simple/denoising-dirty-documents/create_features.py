# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/denoising-dirty-documents/prepared/public"
OUTPUT_DATA_PATH = "output/f88466a1-e032-494a-acbe-a8ee4e4d23cf/9/executor/output"

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

    Standardizes image data for neural network input:
    1. Casts to float32.
    2. Cleans NaN/Inf values.
    3. Enforces [0, 1] normalization (scaling by 255 if necessary).
    4. Ensures (H, W, 1) channel dimension.

    Note:
    - Training/Validation data is expected to be patched (112, 112).
    - Test data is preserved as full images (H, W) but normalized and shaped to (H, W, 1).

    Args:
        X_train (DT): The training set features.
        y_train (DT): The training set labels.
        X_val (DT): The validation set features.
        y_val (DT): The validation set labels.
        X_test (DT): The test set features.

    Returns:
        Tuple[DT, DT, DT, DT, DT]: A tuple containing the transformed data:
        - X_train_transformed
        - y_train_transformed
        - X_val_transformed
        - y_val_transformed
        - X_test_transformed
    """

    def process_dataframe(df: DT, target_col: str) -> DT:
        """
        Helper to process a specific column in a DataFrame or Series containing numpy arrays.
        Applies normalization, cleaning, and reshaping.
        """
        # Handle empty/None inputs
        if df is None:
            return None

        # Create a shallow copy to preserve structure and avoid side effects on original data
        df_transformed = df.copy()

        # Determine if we are working with a Series or DataFrame
        is_dataframe = isinstance(df_transformed, pd.DataFrame)

        if is_dataframe:
            if target_col not in df_transformed.columns:
                # If column not found, return as is (safety fallback)
                return df_transformed
            data_source = df_transformed[target_col]
        else:
            # If input is Series, assume it contains the data directly
            data_source = df_transformed

        # Container for processed arrays
        processed_arrays = []

        # Iterate through the data source
        for arr in data_source:
            # Ensure input is a numpy array
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)

            # 1. Type Conversion
            # Convert to float32 (standard for GPU training)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)

            # 2. Cleaning
            # Replace NaNs with 0 and Infs with boundary values
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

            # 3. Normalization Enforcement
            # If max > 1.0 (with slight tolerance), it suggests [0, 255] range. Normalize to [0, 1].
            # load_data typically returns [0, 1], but this ensures robustness.
            if arr.max() > 1.00001:
                arr = arr / 255.0

            # 4. Clipping
            # Strict enforcement of [0, 1] range to avoid numerical instability
            arr = np.clip(arr, 0.0, 1.0)

            # 5. Reshaping
            # Convert (H, W) to (H, W, 1) for Keras/CNN compatibility
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]

            processed_arrays.append(arr)

        # Assign back transformed data
        if is_dataframe:
            df_transformed[target_col] = processed_arrays
        else:
            df_transformed = pd.Series(processed_arrays, index=df_transformed.index, name=df_transformed.name)

        return df_transformed

    # Apply transformations based on columns defined in load_data.py
    # X contains 'data', y contains 'label'

    X_train_transformed = process_dataframe(X_train, 'data')
    y_train_transformed = process_dataframe(y_train, 'label')

    X_val_transformed = process_dataframe(X_val, 'data')
    y_val_transformed = process_dataframe(y_val, 'label')

    X_test_transformed = process_dataframe(X_test, 'data')

    return X_train_transformed, y_train_transformed, X_val_transformed, y_val_transformed, X_test_transformed

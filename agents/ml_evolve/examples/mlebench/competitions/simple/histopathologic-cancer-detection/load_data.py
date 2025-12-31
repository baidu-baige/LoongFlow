# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/ee5d67f1-b3aa-447c-b45e-155ce5c4f09c/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset for actual training/inference.
            - True: Load a small subset of data (≤100 rows) for quick code validation.

    Returns:
        Tuple[DT, DT, DT, DT]: A tuple containing four elements:
        - X (DT): Training data features (id and file path).
        - y (DT): Training data labels.
        - X_test (DT): Test data features (id and file path).
        - test_ids (DT): Identifiers for the test data.
    """
    # 1. Define File Paths
    train_labels_path = os.path.join(BASE_DATA_PATH, "train_labels.csv")
    sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    train_dir = os.path.join(BASE_DATA_PATH, "train")
    test_dir = os.path.join(BASE_DATA_PATH, "test")

    # 2. Load Tabular Data
    # Read the labels and the submission format (which contains test IDs)
    train_df = pd.read_csv(train_labels_path)
    test_df = pd.read_csv(sample_sub_path)

    # 3. Handle Validation Mode
    if validation_mode:
        train_df = train_df.head(100)
        test_df = test_df.head(100)

    # 4. Map Image Paths
    # Construct the full filesystem path for each image ID
    train_df['path'] = train_df['id'].apply(lambda x: os.path.join(train_dir, f"{x}.tif"))
    test_df['path'] = test_df['id'].apply(lambda x: os.path.join(test_dir, f"{x}.tif"))

    # 5. Verify File Existence
    # This ensures that the generated paths are correct before proceeding to heavy processing
    for path in train_df['path']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training image not found at: {path}")

    for path in test_df['path']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test image not found at: {path}")

    # 6. Prepare Final Outputs
    # Feature columns (X and X_test) must have identical structure
    X = train_df[['id', 'path']]
    y = train_df['label']

    # For X_test, we ensure it matches the column structure of X
    X_test = test_df[['id', 'path']]
    test_ids = test_df['id']

    # Final row alignment check (asserting requirements)
    if len(X) != len(y):
        raise ValueError("Mismatch between training features and labels count.")
    if len(X_test) != len(test_ids):
        raise ValueError("Mismatch between test features and test IDs count.")

    return X, y, X_test, test_ids

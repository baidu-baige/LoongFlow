# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/1799683e-18f2-43a3-97d1-8b0bdddc3200/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset for actual training/inference.
            - True: Load a small subset of data (≤50 rows) for quick code validation.

    Returns:
        Tuple[DT, DT, DT, DT]: A tuple containing four elements:
        - X (DT): Training data features (image paths).
        - y (DT): Training data labels.
        - X_test (DT): Test data features (image paths).
        - test_ids (DT): Identifiers for the test data.
    """
    # Define file paths
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")

    # Load metadata
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Subset data if validation_mode is enabled
    if validation_mode:
        train_df = train_df.head(50)
        test_df = test_df.head(50)

    # Construct full image paths
    # The image_id is the filename stem, extensions are .jpg, located in the images/ directory
    train_df['image_path'] = train_df['image_id'].apply(
        lambda x: os.path.join(BASE_DATA_PATH, "images", f"{x}.jpg")
    )
    test_df['image_path'] = test_df['image_id'].apply(
        lambda x: os.path.join(BASE_DATA_PATH, "images", f"{x}.jpg")
    )

    # Verify image existence (as per requirement)
    for path in train_df['image_path']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Referenced training image not found: {path}")
    for path in test_df['image_path']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Referenced test image not found: {path}")

    # Target columns identified from EDA and requirements: 'healthy', 'multiple_diseases', 'rust', 'scab'
    target_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']

    # Prepare return components
    # Using double brackets for X and X_test to ensure they are returned as DataFrames
    X = train_df[['image_path']]
    y = train_df[target_cols]
    X_test = test_df[['image_path']]
    test_ids = test_df['image_id']

    return X, y, X_test, test_ids

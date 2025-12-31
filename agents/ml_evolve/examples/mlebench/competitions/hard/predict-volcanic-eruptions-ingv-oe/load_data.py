# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Any, Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/43637237-1f49-4750-a868-8602ac177881/1/executor/output"

# Type Definitions
# Semantic type aliases for ML data structures.
Features = Any  # Feature matrix (pd.DataFrame containing segment metadata/paths)
Labels = Any  # Target labels (pd.Series)
TestIDs = Any  # Test set identifiers (pd.Series)


def load_data(validation_mode: bool = False) -> Tuple[Features, Labels, Features, TestIDs]:
    """
    Loads, splits, and returns the initial datasets by providing metadata and 
    file paths for on-the-fly segment processing.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset.
            - True: Load a small subset of data (≤1000 rows) for quick validation.

    Returns:
        Tuple[Features, Labels, Features, TestIDs]:: A tuple containing:
        - X (Features): Training data metadata (segment_id and file paths).
        - y (Labels): Training data eruption times.
        - X_test (Features): Test data metadata (segment_id and file paths).
        - test_ids (TestIDs): Identifiers (segment_id) for the test data.
    """
    # Define paths for metadata files
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")

    # Load metadata (using pandas as it is efficient for these small metadata files)
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Subset data for validation mode if requested
    if validation_mode:
        train_df = train_df.head(1000)
        test_df = test_df.head(1000)

    # Prepare Training Data
    # X stores segment identifiers and full file paths for the feature extractor
    X = pd.DataFrame({
        'segment_id': train_df['segment_id'],
        'path': train_df['segment_id'].apply(
            lambda x: os.path.join(BASE_DATA_PATH, 'train', f"{x}.csv")
        )
    })
    y = train_df['time_to_eruption']

    # Prepare Test Data
    # X_test follows the same structure as X for feature consistency
    X_test = pd.DataFrame({
        'segment_id': test_df['segment_id'],
        'path': test_df['segment_id'].apply(
            lambda x: os.path.join(BASE_DATA_PATH, 'test', f"{x}.csv")
        )
    })
    test_ids = test_df['segment_id']

    # Ensure row alignment and consistency checks pass implicitly by structure
    # Row counts: len(X) == len(y), len(X_test) == len(test_ids)
    # Feature structure: X.columns == X_test.columns

    return X, y, X_test, test_ids

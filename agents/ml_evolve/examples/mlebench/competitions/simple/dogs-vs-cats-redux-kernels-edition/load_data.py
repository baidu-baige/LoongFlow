# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/7dbf3696-f36d-4f87-8e59-bb45d92869c5/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets for the Dogs vs. Cats competition.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset for actual training/inference.
            - True: Load a small subset of data (≤50 rows) for quick code validation.

    Returns:
        Tuple[DT, DT, DT, DT]: A tuple containing four elements:
        - X (DT): Training data features (image paths).
        - y (DT): Training data labels (1 for dog, 0 for cat).
        - X_test (DT): Test data features (image paths).
        - test_ids (DT): Identifiers for the test data.
    """
    train_dir = os.path.join(BASE_DATA_PATH, "train")
    test_dir = os.path.join(BASE_DATA_PATH, "test")

    # 1. Process Training Data
    # Parse labels from filenames: 'dog' -> 1, 'cat' -> 0
    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.jpg')]
    train_records = []
    for f in train_files:
        file_path = os.path.join(train_dir, f)
        label = 1 if 'dog' in f.lower() else 0
        train_records.append({'path': file_path, 'label': label})

    train_df = pd.DataFrame(train_records)

    # 2. Process Test Data
    # Extract numeric ID from filenames like '123.jpg'
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    test_records = []
    for f in test_files:
        file_path = os.path.join(test_dir, f)
        try:
            test_id = int(f.split('.')[0])
            test_records.append({'path': file_path, 'id': test_id})
        except (ValueError, IndexError):
            continue

    test_df = pd.DataFrame(test_records)
    # Ensure test data is sorted by ID for submission consistency
    test_df = test_df.sort_values('id').reset_index(drop=True)

    # 3. Handle Validation Mode
    if validation_mode:
        # Sample a small representative subset (at most 50 rows)
        if not train_df.empty:
            train_df = train_df.sample(n=min(50, len(train_df)), random_state=42).reset_index(drop=True)
        if not test_df.empty:
            # We sample then re-sort to maintain ID order in validation output
            test_df = test_df.sample(n=min(50, len(test_df)), random_state=42).sort_values('id').reset_index(drop=True)

    # 4. Format returns
    X = train_df[['path']]
    y = train_df['label']
    X_test = test_df[['path']]
    test_ids = test_df['id']

    return X, y, X_test, test_ids

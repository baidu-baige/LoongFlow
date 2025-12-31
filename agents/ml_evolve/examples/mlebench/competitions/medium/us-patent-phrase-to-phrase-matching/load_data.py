# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import cudf
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/aefcc010-8f21-4ecb-b149-7bf99579e6d3/6/executor/output"

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
        - X (DT): Training data features.
        - y (DT): Training data labels.
        - X_test (DT): Test data features.
        - test_ids (DT): Identifiers for the test data.
    """
    # Determine the number of rows to load
    nrows = 100 if validation_mode else None

    # Define file paths
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")

    # Load data using cudf for GPU acceleration
    train_df = cudf.read_csv(train_path, nrows=nrows)
    test_df = cudf.read_csv(test_path, nrows=nrows)

    # CPC Section mapping
    cpc_map = {
        'A': 'Human Necessities',
        'B': 'Performing Operations; Transporting',
        'C': 'Chemistry; Metallurgy',
        'D': 'Textiles; Paper',
        'E': 'Fixed Constructions',
        'F': 'Mechanical Engineering; Lighting; Heating; Weapons; Blasting',
        'G': 'Physics',
        'H': 'Electricity'
    }

    # Use a cudf Series to perform the mapping on GPU
    map_ser = cudf.Series(cpc_map)

    # Create the context_desc feature: title + space + full code
    # e.g., 'H04' -> 'Electricity H04'
    train_df['context_desc'] = train_df['context'].str[0].map(map_ser) + " " + train_df['context']
    test_df['context_desc'] = test_df['context'].str[0].map(map_ser) + " " + test_df['context']

    # Separate features and target from the training set
    # Expected columns: [id, anchor, target, context, score, context_desc]
    y = train_df['score'].to_pandas()
    X = train_df.drop(columns=['score']).to_pandas()

    # Prepare test data features and identifiers
    # Expected columns: [id, anchor, target, context, context_desc]
    X_test = test_df.to_pandas()
    test_ids = test_df['id'].to_pandas()

    # Validation: Ensure all return values are non-empty and row-aligned
    if X.empty or y.empty or X_test.empty or test_ids.empty:
        raise ValueError("One or more returned datasets are empty.")

    if len(X) != len(y):
        raise ValueError(f"X and y row count mismatch: {len(X)} != {len(y)}")

    if len(X_test) != len(test_ids):
        raise ValueError(f"X_test and test_ids row count mismatch: {len(X_test)} != {len(test_ids)}")

    # Ensure X and X_test have identical feature structure
    # Both now contain: id, anchor, target, context, context_desc

    return X, y, X_test, test_ids

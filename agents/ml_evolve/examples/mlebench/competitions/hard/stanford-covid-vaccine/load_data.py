# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/832d0196-b83e-4fa9-8ea2-3588ff903a43/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets for the RNA degradation prediction task.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset.
            - True: Load a small subset (≤100 rows) for quick validation.

    Returns:
        Tuple[DT, DT, DT, DT]: 
        - X (pd.DataFrame): Training features including sequences and metadata.
        - y (pd.DataFrame): Training targets (reactivity and degradation rates).
        - X_test (pd.DataFrame): Test features with identical structure to X.
        - test_ids (pd.Series): Identifiers for the test data samples.
    """
    train_path = os.path.join(BASE_DATA_PATH, 'train.json')
    test_path = os.path.join(BASE_DATA_PATH, 'test.json')

    # Load JSON data (lines=True as specified in dataset description)
    # Using pandas as it robustly handles nested lists in JSON-lines format
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    # Validation mode subsetting
    if validation_mode:
        train_df = train_df.head(100)
        test_df = test_df.head(100)

    # 5 target columns to be predicted
    target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    # Feature columns to retain for modeling and weighting
    # We include seq_length/seq_scored to handle the variable lengths in the test set
    feature_cols = [
        'id', 'sequence', 'structure', 'predicted_loop_type',
        'seq_length', 'seq_scored', 'signal_to_noise', 'SN_filter'
    ]

    # Extract targets
    y = train_df[target_cols].copy()

    # Extract training features
    X = train_df[feature_cols].copy()

    # Prepare test features with identical structure
    # test.json typically lacks signal_to_noise and SN_filter; we fill them with defaults
    X_test = test_df.reindex(columns=feature_cols)
    X_test['signal_to_noise'] = X_test['signal_to_noise'].fillna(0.0)
    X_test['SN_filter'] = X_test['SN_filter'].fillna(0)

    # Extract test identifiers
    test_ids = test_df['id'].copy()

    # Final validation of requirements
    if X.empty or y.empty or X_test.empty or test_ids.empty:
        raise ValueError("One or more returned datasets are empty.")

    if len(X) != len(y):
        raise ValueError(f"Mismatch in training samples: X({len(X)}) vs y({len(y)})")

    if len(X_test) != len(test_ids):
        raise ValueError(f"Mismatch in test samples: X_test({len(X_test)}) vs test_ids({len(test_ids)})")

    return X, y, X_test, test_ids

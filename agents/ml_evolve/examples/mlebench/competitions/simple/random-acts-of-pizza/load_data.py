# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import json
import os
from collections import Counter
from typing import Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux-ml/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/ed330620-ed29-4387-b009-fed5bf45c1a8/11/executor/output"

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
        - X (DT): Training data features.
        - y (DT): Training data labels.
        - X_test (DT): Test data features.
        - test_ids (DT): Identifiers for the test data.
    """
    # Load training data
    train_path = os.path.join(BASE_DATA_PATH, "train.json")
    with open(train_path, 'r') as f:
        train_data = json.load(f)

    # Load test data
    test_path = os.path.join(BASE_DATA_PATH, "test.json")
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Handle validation mode
    if validation_mode:
        # For training data, sample up to 50 rows while maintaining class balance
        # Using 25 samples per class (as per EDA, classes are imbalanced)
        train_df = train_df.groupby('requester_received_pizza', group_keys=False) \
            .apply(lambda x: x.sample(min(len(x), 25), random_state=42)) \
            .sample(frac=1, random_state=42) \
            .head(50)

        # For test data, just take the first 50 rows (no labels available)
        test_df = test_df.head(50)

    # Split training data into features and target
    X = train_df.drop(columns=['requester_received_pizza'])
    y = train_df['requester_received_pizza']

    # Prepare test data
    X_test = test_df.copy()
    test_ids = test_df['request_id']

    # Handle missing values in requester_user_flair (75% missing per EDA)
    if 'requester_user_flair' in X.columns:
        most_common_flair = Counter(X['requester_user_flair'].dropna()).most_common(1)[0][0]
        X['requester_user_flair'].fillna(most_common_flair, inplace=True)
        if 'requester_user_flair' in X_test.columns:
            X_test['requester_user_flair'].fillna(most_common_flair, inplace=True)

    # Convert unix timestamps to datetime (per EDA, these are highly correlated)
    timestamp_cols = ['unix_timestamp_of_request', 'unix_timestamp_of_request_utc']
    for col in timestamp_cols:
        if col in X.columns:
            X[col] = pd.to_datetime(X[col], unit='s')
        if col in X_test.columns:
            X_test[col] = pd.to_datetime(X_test[col], unit='s')

    # Ensure column alignment between X and X_test
    # Get intersection of columns (excluding target)
    common_cols = list(set(X.columns) & set(X_test.columns))
    X = X[common_cols]
    X_test = X_test[common_cols]

    # Ensure same column order (important for some models)
    X_test = X_test[X.columns]

    # Verify all return values are non-empty
    assert len(X) > 0 and len(y) > 0 and len(X_test) > 0 and len(test_ids) > 0
    # Verify row alignment
    assert len(X) == len(y) and len(X_test) == len(test_ids)
    # Verify column alignment
    assert len(X.columns) == len(X_test.columns)
    assert list(X.columns) == list(X_test.columns)

    return X, y, X_test, test_ids

# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/bb8b0571-64fb-49b6-8e49-1fc2d52da49b/2/executor/output"

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
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")

    nrows = 50 if validation_mode else None

    # Load datasets
    df_train = pd.read_csv(train_path, nrows=nrows)
    df_test = pd.read_csv(test_path, nrows=nrows)

    def decode_comment(s):
        """Handles unicode-escaped text sequences in the Comment column."""
        if isinstance(s, str):
            try:
                # Convert literal escape sequences (e.g., \xNN, \uNNNN) into actual characters
                return s.encode('latin-1').decode('unicode_escape')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Propagate or return as-is if it's already properly formatted or contains complex chars
                return s
        return s

    # Preprocess Training Data
    if 'Comment' in df_train.columns:
        df_train['Comment'] = df_train['Comment'].apply(decode_comment)
    if 'Date' in df_train.columns:
        df_train['Date'] = pd.to_datetime(df_train['Date'], format='%Y%m%d%H%M%SZ', errors='coerce')

    # Preprocess Test Data
    if 'Comment' in df_test.columns:
        df_test['Comment'] = df_test['Comment'].apply(decode_comment)
    if 'Date' in df_test.columns:
        df_test['Date'] = pd.to_datetime(df_test['Date'], format='%Y%m%d%H%M%SZ', errors='coerce')

    # Extract features and targets
    # Based on dataset description: train has Insult, Date, Comment. Test has Date, Comment.
    y = df_train['Insult']
    X = df_train[['Date', 'Comment']]
    X_test = df_test[['Date', 'Comment']]

    # test_ids: Use the index as the identifier since no explicit ID column is provided in test.csv
    test_ids = pd.Series(df_test.index, name='id')

    return X, y, X_test, test_ids

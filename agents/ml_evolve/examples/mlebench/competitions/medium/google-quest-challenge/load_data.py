# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
import re
from typing import Tuple

import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/ea4f9f02-c6be-4ad0-a6fd-71930e8fb81e/15/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets for the Google QUEST challenge.
    
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
    train_path = os.path.join(BASE_DATA_PATH, 'train.csv')
    test_path = os.path.join(BASE_DATA_PATH, 'test.csv')
    sub_path = os.path.join(BASE_DATA_PATH, 'sample_submission.csv')

    # Identify target columns from sample submission to ensure correct labels and order
    sub = pd.read_csv(sub_path)
    target_cols = [c for c in sub.columns if c != 'qa_id']

    # Load raw datasets
    nrows = 100 if validation_mode else None
    train = pd.read_csv(train_path, nrows=nrows)
    test = pd.read_csv(test_path, nrows=nrows)

    # Feature columns are those present in the test set
    feature_cols = list(test.columns)

    def clean_text(text):
        """
        Normalizes whitespace and collapses multiple newlines into a single \n.
        """
        if pd.isna(text):
            return ""
        text = str(text)
        # Collapse multiple newline characters into a single \n
        text = re.sub(r'\n+', '\n', text)
        # Normalize other whitespace (tabs, multiple spaces) to a single space
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    # Apply cleaning to primary text columns
    text_cols = ['question_title', 'question_body', 'answer']
    for col in text_cols:
        if col in train.columns:
            train[col] = train[col].astype(str).apply(clean_text)
        if col in test.columns:
            test[col] = test[col].astype(str).apply(clean_text)

    # Extract features, labels and IDs
    X = train[feature_cols]
    y = train[target_cols]
    X_test = test[feature_cols]
    test_ids = test['qa_id']

    # Integrity Checks - Propagate errors if data structures are invalid
    if X.empty or y.empty or X_test.empty or test_ids.empty:
        raise ValueError("Data loading resulted in one or more empty structures.")

    if len(X) != len(y):
        raise ValueError(f"Train features and labels row mismatch: {len(X)} vs {len(y)}")

    if len(X_test) != len(test_ids):
        raise ValueError(f"Test features and IDs row mismatch: {len(X_test)} vs {len(test_ids)}")

    if list(X.columns) != list(X_test.columns):
        raise ValueError("Feature columns in X and X_test are not identical.")

    return X, y, X_test, test_ids

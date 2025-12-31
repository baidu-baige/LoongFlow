# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/40d01db3-cd9d-46e2-8d9c-d192fb8addff/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets for the Jigsaw Toxic Comment Classification Challenge.
    """
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")

    nrows = 100 if validation_mode else None

    # Load data
    train_df = pd.read_csv(train_path, nrows=nrows)
    test_df = pd.read_csv(test_path, nrows=nrows)

    # Define columns
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    feature_col = 'comment_text'
    id_col = 'id'

    # Prepare training features and labels
    # We keep it as a DataFrame to maintain feature structure consistency
    X = train_df[[feature_col]]
    y = train_df[target_cols].astype(np.float32)

    # Prepare test features and ids
    X_test = test_df[[feature_col]]
    test_ids = test_df[id_col]

    # Verification of requirements
    if X.empty or y.empty or X_test.empty or test_ids.empty:
        raise ValueError("One or more loaded datasets are empty.")

    if len(X) != len(y):
        raise ValueError(f"Alignment error: X has {len(X)} rows but y has {len(y)} rows.")

    if len(X_test) != len(test_ids):
        raise ValueError(f"Alignment error: X_test has {len(X_test)} rows but test_ids has {len(test_ids)} rows.")

    if list(X.columns) != list(X_test.columns):
        raise ValueError("Feature consistency error: X and X_test columns do not match.")

    return X, y, X_test, test_ids

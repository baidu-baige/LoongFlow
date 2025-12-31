# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/aerial-cactus-identification/prepared/public"
OUTPUT_DATA_PATH = "output/153e4624-940b-4d19-a37d-90435531bfd1/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.

    Args:
        X (DT): The full training data features. Useful for checking if 'Group' or 'Time' columns exist.
        y (DT): The full training data labels. Useful for checking class distribution (Stratified).

    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter 
                            (e.g., KFold, StratifiedKFold, TimeSeriesSplit).
    """
    # Step 1: Analyze the task type
    # This is a binary classification task (has_cactus: 0 or 1)
    # Class distribution is imbalanced: ~75% class 1, ~25% class 0

    # Step 2: Check for special constraints
    # No time-series ordering (images are independent)
    # No group structure (each image is independent, no patient/user IDs)

    # Step 3: Instantiate StratifiedKFold
    # - n_splits=5: Standard choice, provides good balance between bias and variance
    # - shuffle=True: Randomize the data before splitting to avoid ordering bias
    # - random_state=42: Fixed seed for reproducibility
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 4: Return the splitter object
    return cv

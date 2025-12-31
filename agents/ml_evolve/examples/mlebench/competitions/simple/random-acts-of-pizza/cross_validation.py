# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux-ml/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/ed330620-ed29-4387-b009-fed5bf45c1a8/11/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
    
    For this classification task with imbalanced classes (75% negative, 25% positive),
    we use StratifiedKFold to maintain the class distribution in each fold.
    
    Args:
        X (DT): The full training data features. Not used directly here but useful for
                checking additional constraints in more complex scenarios.
        y (DT): The full training data labels. Used to verify classification task and
                class imbalance.
        
    Returns:
        BaseCrossValidator: A StratifiedKFold splitter with 5 folds, shuffling enabled,
                          and fixed random state for reproducibility.
    """
    # Using StratifiedKFold with 5 folds for robust validation
    # shuffle=True to randomize the data before splitting (important for time-related features)
    # random_state=42 for reproducibility
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return cv

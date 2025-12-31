# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/ee5d67f1-b3aa-447c-b45e-155ce5c4f09c/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
    
    Args:
        X (DT): The full training data features. 
        y (DT): The full training data labels.
    
    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter (StratifiedKFold).
    """
    # Based on the task description and EDA, the dataset contains binary labels 
    # with a ~40.5% positive class distribution. 
    # StratifiedKFold ensures that each fold maintains this label proportion,
    # which is critical for stable performance estimation (AUC) in classification tasks.

    # Configuration:
    # n_splits=5: Common choice providing a good balance between bias and variance.
    # shuffle=True: Necessary since the data might have some ordering.
    # random_state=42: Ensures reproducibility across runs.

    cv_splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    return cv_splitter

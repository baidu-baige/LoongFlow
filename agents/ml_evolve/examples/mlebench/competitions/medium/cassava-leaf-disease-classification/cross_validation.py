# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/068f0c14-e630-462e-bc46-9d2d4b1d5fc3/5/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.

    Args:
        X (DT): The full training data features. 
        y (DT): The full training data labels.

    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter.

    Requirements:   
      - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
    """
    # Step 1: Analyze task type and data characteristics
    # The task is a multi-class classification problem (5 classes) with significant class imbalance.
    # Class 3 (Cassava Mosaic Disease) is the majority class.

    # Step 2: Select appropriate splitter based on analysis
    # StratifiedKFold ensures that each fold maintains the same proportion of labels as the original dataset.
    # We use 5 folds, shuffle the data, and set a fixed random state for reproducibility.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Step 3: Return configured splitter instance
    return cv

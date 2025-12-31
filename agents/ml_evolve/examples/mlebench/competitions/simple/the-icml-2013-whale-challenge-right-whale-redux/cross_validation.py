# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/5872ea81-cdbf-4bc6-8da8-7d0e82d40021/2/executor/output"

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
    # The dataset has a significant class imbalance (10% whale calls).
    # Stratified CV is essential to maintain representative distributions across folds.

    # Step 2: Select appropriate splitter based on analysis
    # Stratified 5-Fold Cross-Validation ensures each fold has the same proportion of classes as the original dataset.
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Step 3: Return configured splitter instance
    return cv_splitter

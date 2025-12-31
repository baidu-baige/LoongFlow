# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import BaseCrossValidator

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/40d01db3-cd9d-46e2-8d9c-d192fb8addff/1/executor/output"

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
        - Use MultilabelStratifiedKFold for robust evaluation across imbalanced multi-labels.
        - 5-fold CV with shuffle enabled and seed 42.
    """
    # Step 1: Analyze task type and data characteristics
    # The task is multi-label classification with significant class imbalance (e.g., 'threat' at 0.3%).
    # Standard K-Fold or StratifiedKFold (single-label) would not maintain the label proportions
    # across the multi-label space.

    # Step 2: Select appropriate splitter based on analysis
    # MultilabelStratifiedKFold from iterative-stratification is the standard choice 
    # for maintaining consistent label distributions across all target columns.
    cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Step 3: Return configured splitter instance
    # Downstream components will call cv.split(X, y) where y contains all 6 target columns,
    # ensuring the stratification objective is met.
    return cv

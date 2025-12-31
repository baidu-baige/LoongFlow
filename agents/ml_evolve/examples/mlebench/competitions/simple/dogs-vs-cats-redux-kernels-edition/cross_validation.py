# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/7dbf3696-f36d-4f87-8e59-bb45d92869c5/1/executor/output"

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
    # The task is binary classification (cat vs dog). The EDA indicates a balanced dataset.

    # Step 2: Select appropriate splitter based on analysis
    # StratifiedKFold is used to ensure each fold has the same proportion of classes as the original dataset.
    # n_splits=5, shuffle=True, random_state=42 as per implementation guidance.
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Step 3: Return configured splitter instance
    return splitter

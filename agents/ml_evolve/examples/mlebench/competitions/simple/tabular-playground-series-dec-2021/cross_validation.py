# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/ead961ce-50a1-41ec-89e9-4cc0d527fbe5/1/executor/output"

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
    # The task is multi-class classification on a large imbalanced dataset.
    # Class 5 has been removed in the load_data step, ensuring all remaining classes 
    # have enough samples for at least a few folds (Class 4 has 333 samples).

    # Step 2: Select appropriate splitter based on analysis
    # StratifiedKFold is ideal for classification tasks with imbalanced classes to 
    # ensure each fold reflects the overall class distribution.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 3: Return configured splitter instance
    return splitter

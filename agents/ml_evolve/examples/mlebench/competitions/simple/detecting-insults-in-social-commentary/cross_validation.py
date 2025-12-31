# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/bb8b0571-64fb-49b6-8e49-1fc2d52da49b/2/executor/output"

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
    # The dataset is relatively small (3947 rows) and imbalanced (approx. 27% insults).
    # The evaluation metric is AUC.

    # Step 2: Select appropriate splitter based on analysis
    # StratifiedKFold is selected to ensure that each fold maintains the same proportion
    # of the 'insult' class as the full training set. This is critical for reliable
    # AUC measurement and model generalization on imbalanced data.
    # n_splits=5 provides a balance between having enough training data in each fold
    # and enough folds to estimate performance variance.
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Step 3: Return configured splitter instance
    return splitter

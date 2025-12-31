# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/832d0196-b83e-4fa9-8ea2-3588ff903a43/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class SNFilterStratifiedKFold(StratifiedKFold):
    """
    Custom StratifiedKFold that ensures stratification is performed on the 'SN_filter' 
    column from the input feature set X. This ensures each fold has a representative 
    proportion of high-quality samples (SN_filter=1), which is critical given the 
    noise-to-signal characteristics of the dataset.
    """

    def split(self, X: DT, y: DT = None, groups: Any = None):
        """
        Generates indices to split data into training and test set.
        
        Args:
            X (DT): The features. Must contain 'SN_filter' column.
            y (DT): The target labels (ignored for stratification).
            groups (Any): Group labels (ignored).
        """
        # Stratify on the 'SN_filter' column as specified in the strategy requirements
        return super().split(X, X['SN_filter'], groups)


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
    
    Args:
        X (DT): The full training data features. 
        y (DT): The full training data labels.
    
    Returns:
        BaseCrossValidator: An instance of a 5-fold Stratified K-Fold splitter 
                            stratified by 'SN_filter'.
    """
    # Strategy: 5-fold Stratified K-Fold.
    # Stratification Column: SN_filter.
    # Metric: MCRMSE (calculated during training/evaluation).
    # Splitting: Split at the molecule (row) level.

    # We use a custom splitter class to bake in the SN_filter stratification 
    # requirement, ensuring it is used even if the standard split(X, y) is called 
    # where y contains the multi-output continuous targets.

    return SNFilterStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

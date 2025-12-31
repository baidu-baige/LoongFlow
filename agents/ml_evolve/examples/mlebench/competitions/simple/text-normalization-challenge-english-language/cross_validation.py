# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedGroupKFold

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/text-normalization-challenge-english-language/prepared/public"
OUTPUT_DATA_PATH = "output/12e29d80-a70c-426a-9331-de3aa1a6ce7c/14/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class StratifiedGroupKFoldByClass(StratifiedGroupKFold):
    """
    A specialized StratifiedGroupKFold that handles the specific structure 
    of the text normalization dataset. It stratifies by the 'class' column 
    and groups by 'sentence_id' to ensure that tokens from the same sentence 
    stay together in the same fold while maintaining class balance.
    """

    def split(self, X: DT, y: DT, groups: Any = None):
        """
        Overrides split to ensure stratification is performed on the 'class' column
        and grouping is performed on 'sentence_id'.
        
        Args:
            X (DT): The full training data features.
            y (DT): The full training data labels, containing the 'class' column.
            groups (Any): Ignored here as 'sentence_id' is extracted from X.
        """
        # Extract 'class' for stratification from y. 
        # If 'class' is missing, the error will propagate as required.
        y_stratify = y['class']

        # Extract 'sentence_id' for grouping from X.
        group_col = X['sentence_id']

        # Propagate to parent implementation which handles the actual fold logic
        return super().split(X, y_stratify, groups=group_col)


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
    
    Args:
        X (DT): The full training data features. 
        y (DT): The full training data labels.
    
    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter.
    """
    # Initialize the custom splitter with 5 folds.
    # shuffle=True with fixed random_state for reproducibility.
    # This strategy groups by sentence_id to avoid data leakage (sentence integrity)
    # and stratifies by token class to maintain stable distribution across folds.
    splitter = StratifiedGroupKFoldByClass(n_splits=5, shuffle=True, random_state=42)

    return splitter

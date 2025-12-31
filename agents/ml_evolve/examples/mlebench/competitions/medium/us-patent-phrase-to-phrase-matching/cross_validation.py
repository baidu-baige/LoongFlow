# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, GroupKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/aefcc010-8f21-4ecb-b149-7bf99579e6d3/6/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class AnchorGroupKFold(GroupKFold):
    """
    Custom GroupKFold that automatically uses the 'anchor' column as groups
    if the groups parameter is not provided during the split call.
    """

    def split(self, X, y=None, groups=None):
        if groups is None:
            if isinstance(X, pd.DataFrame) and 'anchor' in X.columns:
                groups = X['anchor']
            else:
                # Let error propagate as per requirements
                raise ValueError("The 'groups' parameter is None and 'anchor' column was not found in X.")
        return super().split(X, y, groups)


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
  
    Args:
        X (DT): The full training data features. 
        y (DT): The full training data labels.
  
    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter.
    """
    # Step 1: Initialize the GroupKFold splitter with 5 splits as per parent logic.
    # Grouping by 'anchor' ensures that all phrase pairs sharing the same anchor 
    # are kept in the same fold, preventing data leakage and testing generalization to new anchors.
    cv = AnchorGroupKFold(n_splits=5)

    # Step 2: Validate presence of 'anchor' column
    if 'anchor' not in X.columns:
        raise KeyError("Required column 'anchor' not found in features X for GroupKFold.")

    # Step 3: Assign fold index to the training dataframe for downstream model tracking.
    groups = X['anchor']
    X['fold'] = -1

    # Iterate through the folds to assign the fold index to each sample.
    # Using X.index[val_idx] ensures row alignment regardless of index state.
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        X.loc[X.index[val_idx], 'fold'] = fold_idx

    # Step 4: Return configured splitter instance
    return cv

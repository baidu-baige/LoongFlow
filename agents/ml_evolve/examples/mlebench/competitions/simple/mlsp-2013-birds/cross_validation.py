# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/b4d4dfce-4367-41a9-8cac-a59279a6d65f/11/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
    
    Strategies:
    1. IterativeStratifiedKFold: Best for multi-label data (tries to preserve label ratios).
    2. LabelPowersetStratifiedKFold (Fallback): Treats unique label combinations as classes for Stratification.
    3. KFold (Final Fallback): Standard random splitting if stratification fails (e.g. rare singletons).

    Args:
        X (DT): The full training data features.
        y (DT): The full training data labels.

    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter.
    """

    # Configuration
    N_SPLITS = 5
    RANDOM_STATE = 42
    SHUFFLE = True

    # -------------------------------------------------------------------------
    # Strategy 1: IterativeStratifiedKFold (Preferred)
    # -------------------------------------------------------------------------
    # We attempt to import from known libraries that support multi-label stratification
    try:
        from iterstrat.ml_stratifiers import IterativeStratifiedKFold
        # order=1 attempts to balance label pairs if possible, order=2 for triplets, etc.
        # order=1 is a good balance of speed and performance.
        return IterativeStratifiedKFold(n_splits=N_SPLITS, order=1, random_state=RANDOM_STATE)
    except (ImportError, TypeError):
        pass

    try:
        from skmultilearn.model_selection import IterativeStratifiedKFold
        return IterativeStratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)
    except (ImportError, TypeError):
        pass

    # -------------------------------------------------------------------------
    # Strategy 2: Custom Label Powerset Stratification (Fallback)
    # -------------------------------------------------------------------------
    class LabelPowersetStratifiedKFold(BaseCrossValidator):
        """
        Custom CrossValidator that performs Stratified K-Fold on the 
        'Label Powerset' (unique combinations of labels) for multi-label data.
        Falls back to KFold if stratification fails (e.g. singletons).
        """

        def __init__(self, n_splits: int, shuffle: bool = True, random_state: int = None):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state
            self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            """
            Generate indices to split data into training and test set.
            """
            if y is None:
                yield from self.kf.split(X, y, groups)
                return

            # Ensure consistent numpy array format
            if hasattr(y, "to_numpy"):
                y_arr = y.to_numpy()
            else:
                y_arr = np.array(y)

            # Detect multi-label vs single-label
            # Multi-label is typically 2D with >1 columns
            is_multilabel = y_arr.ndim > 1 and y_arr.shape[1] > 1

            if is_multilabel:
                # Powerset method: Convert label vector [0, 1, 0] to string "010"
                # This treats every unique combination as a distinct class for stratification
                # Ensuring int conversion prevents '0.0' float string issues
                y_powerset = np.array(["".join(row.astype(int).astype(str)) for row in y_arr])

                try:
                    yield from self.skf.split(X, y_powerset, groups)
                except ValueError:
                    # StratifiedKFold fails if a class has fewer members than n_splits.
                    # With small datasets and many labels, unique combinations (singletons) are common.
                    # Fallback to pure random KFold in this edge case to prevent crash.
                    yield from self.kf.split(X, y_arr, groups)
            else:
                # Standard StratifiedKFold for single-label/binary
                try:
                    # Ensure 1D for StratifiedKFold
                    y_target = y_arr.ravel() if y_arr.ndim > 1 else y_arr
                    yield from self.skf.split(X, y_target, groups)
                except ValueError:
                    # Fallback for extreme imbalance in single-label too
                    yield from self.kf.split(X, y_arr, groups)

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

    return LabelPowersetStratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)

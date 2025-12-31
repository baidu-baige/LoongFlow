# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, GroupKFold, KFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/denoising-dirty-documents/prepared/public"
OUTPUT_DATA_PATH = "output/f88466a1-e032-494a-acbe-a8ee4e4d23cf/9/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.

    Strategy:
    1. Primary: Use GroupKFold (n_splits=5) grouping by 'image_id'. 
       This prevents data leakage by ensuring all overlapping patches from the same 
       source document reside exclusively in either the training or validation set.
    2. Fallback: If 'image_id' is missing or the number of unique images is insufficient 
       (e.g., during low-data debugging/validation runs), adaptively reduce splits 
       or fall back to KFold to ensure the pipeline continues to execute.

    Args:
        X (DT): The full training data features. Expected to contain 'image_id'.
        y (DT): The full training data labels.

    Returns:
        BaseCrossValidator: An instance of the custom ImageIdGroupKFold splitter.
    """

    class ImageIdGroupKFold(BaseCrossValidator):
        """
        Custom CrossValidator that wraps GroupKFold to automatically use 'image_id' 
        from the input DataFrame as the grouping key.
        """

        def __init__(self, n_splits: int = 5, group_col: str = 'image_id'):
            self.n_splits = n_splits
            self.group_col = group_col

        def _get_groups(self, X):
            """Helper to extract groups from X if it's a DataFrame."""
            if isinstance(X, pd.DataFrame) and self.group_col in X.columns:
                return X[self.group_col]
            return None

        def split(self, X, y=None, groups=None):
            """
            Generate indices to split data into training and test set.
            """
            # 1. Resolve Groups
            # If groups are not explicitly passed, try to extract from X['image_id']
            if groups is None:
                groups = self._get_groups(X)

            # 2. Analyze Group Availability & Choose Strategy
            if groups is not None:
                # Get number of unique groups (documents)
                if hasattr(groups, 'unique'):
                    unique_groups = groups.unique()
                else:
                    unique_groups = np.unique(groups)

                n_groups = len(unique_groups)

                # Case A: Sufficient groups for requested splits (Standard)
                if n_groups >= self.n_splits:
                    yield from GroupKFold(n_splits=self.n_splits).split(X, y, groups=groups)
                    return

                # Case B: Limited groups (e.g. Validation Mode with 2-4 images)
                # Reduce n_splits to match n_groups to avoid errors
                elif n_groups >= 2:
                    yield from GroupKFold(n_splits=n_groups).split(X, y, groups=groups)
                    return

                # Case C: Single group (1 image) -> Fall through to KFold

            # 3. Fallback Strategy (Leakage Warning)
            # Used when:
            # - No 'image_id' column found
            # - Only 1 image is loaded (cannot do group split)
            # We switch to KFold on the patches directly.

            n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
            effective_splits = min(self.n_splits, n_samples)

            if effective_splits >= 2:
                # Use shuffle=True with fixed seed for reproducibility in fallback
                yield from KFold(n_splits=effective_splits, shuffle=True, random_state=42).split(X, y)

        def get_n_splits(self, X=None, y=None, groups=None):
            """Returns the number of splitting iterations."""
            if groups is None:
                groups = self._get_groups(X)

            if groups is not None:
                if hasattr(groups, 'unique'):
                    n_groups = len(groups.unique())
                else:
                    n_groups = len(np.unique(groups))

                if n_groups >= self.n_splits:
                    return self.n_splits
                elif n_groups >= 2:
                    return n_groups

            # Fallback KFold logic
            if X is not None:
                n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
                return min(self.n_splits, n_samples)
            return self.n_splits

    # Return the configured custom splitter
    return ImageIdGroupKFold(n_splits=5)

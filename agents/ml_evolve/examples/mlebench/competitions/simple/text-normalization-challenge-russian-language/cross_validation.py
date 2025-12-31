# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, GroupKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/dce1a922-fb4b-4006-9d03-6f53b7ea0718/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class SentenceGroupKFold(BaseCrossValidator):
    """
    Custom cross-validator that wraps GroupKFold to use sentence_id as groups.
    This ensures tokens from the same sentence stay together in the same fold,
    preventing data leakage.
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self._group_kfold = GroupKFold(n_splits=n_splits)
        self._groups = None

    def set_groups(self, groups):
        """Store groups for later use in split."""
        self._groups = groups

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data features (DataFrame with sentence_id column)
            y: Target variable (optional)
            groups: Group labels for the samples (if None, uses sentence_id from X)
        
        Yields:
            train_idx, test_idx: Indices for training and test sets
        """
        # Determine groups - use provided groups or extract from X
        if groups is None:
            if self._groups is not None:
                groups = self._groups
            elif isinstance(X, pd.DataFrame) and 'sentence_id' in X.columns:
                groups = X['sentence_id'].values
            else:
                raise ValueError("Groups must be provided or X must contain 'sentence_id' column")

        # Use GroupKFold to split by sentence_id
        for train_idx, test_idx in self._group_kfold.split(X, y, groups):
            yield train_idx, test_idx


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.

    Args:
        X (DT): The full training data features. Contains 'sentence_id' for grouping.
        y (DT): The full training data labels. Contains normalized text targets.

    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter 
                            using GroupKFold with sentence_id as groups.
    """
    # Step 1: Create the custom cross-validator with sentence-level grouping
    # Using n_splits=5 as specified in the guidance
    cv = SentenceGroupKFold(n_splits=5, random_state=42)

    # Step 2: Extract groups from X if available
    if isinstance(X, pd.DataFrame) and 'sentence_id' in X.columns:
        groups = X['sentence_id'].values
        cv.set_groups(groups)

        # Log some statistics about the grouping
        n_sentences = len(np.unique(groups))
        n_tokens = len(X)
        print(f"Cross-validation setup:")
        print(f"  - Number of folds: 5")
        print(f"  - Total tokens: {n_tokens:,}")
        print(f"  - Total sentences (groups): {n_sentences:,}")
        print(f"  - Average tokens per sentence: {n_tokens / n_sentences:.2f}")
        print(f"  - Grouping by: sentence_id (keeps sentences intact)")

        # If class information is available, show distribution
        if 'class' in X.columns:
            class_dist = X['class'].value_counts()
            print(f"  - Number of token classes: {len(class_dist)}")
            print(f"  - Top 5 classes: {dict(class_dist.head())}")

    # Step 3: Return the cross-validator
    return cv

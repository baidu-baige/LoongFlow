# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/1ae93cb9-976c-4242-a6ab-6fcbc92f5502/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class StratifiedLabelSplitter(BaseCrossValidator):
    """
    Custom cross-validation splitter that stratifies based on the unique 
    combinations of labels found in the original dataset.
    """

    def __init__(self, y_stratify: pd.Series, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.y_stratify = y_stratify
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        """
        Generates indices to split data into training and test set.
        Overrides the default behavior to ensure stratification on the original 
        multi-label combinations (y_stratify) rather than the binarized y.
        """
        return self.skf.split(X, self.y_stratify)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
    
    Args:
        X (DT): The full training data features (containing 'image_path'). 
        y (DT): The full training data labels (binarized).
    
    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter.
    """
    # Load original training metadata to retrieve the 'labels' string column
    # This column contains the raw combinations (e.g., 'scab frog_eye_leaf_spot')
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    # Create a mapping from image ID to original labels string
    label_map = dict(zip(train_df['image'], train_df['labels']))

    # Extract the labels corresponding to the current training set X
    # X['image_path'] contains filenames as defined in the load_data component
    y_stratify = X['image_path'].map(label_map)

    # Verification: Ensure every image in X has a corresponding label entry
    if y_stratify.isnull().any():
        raise ValueError("Missing labels in train.csv for one or more images in the featureset X.")

    # Return a 5-fold Stratified K-Fold strategy
    # Stratifying on y_stratify preserves the distribution of the 12 unique label combinations
    return StratifiedLabelSplitter(
        y_stratify=y_stratify,
        n_splits=5,
        shuffle=True,
        random_state=42
    )

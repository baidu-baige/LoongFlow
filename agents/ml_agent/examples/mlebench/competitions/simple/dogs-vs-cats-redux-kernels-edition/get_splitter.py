from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/25e7371d-bfe6-47c9-b200-bdf664ef9932/2/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame   # Feature matrix type (contains paths and metadata)
y = pd.Series      # Target vector type (binary labels)

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    This function utilizes StratifiedKFold to ensure that each of the 5 folds 
    maintains the 50/50 class distribution of cats and dogs, providing stable
    Out-of-Fold (OOF) estimates for Log Loss evaluation.

    Args:
        X (X): The full training features (paths and IDs).
        y (y): The full training targets (0 for cat, 1 for dog).

    Returns:
        StratifiedKFold: A splitter object configured for 5-fold cross-validation.
    """
    # Step 1: Initialize the StratifiedKFold splitter with specified parameters
    # Requirement: n_splits=5, shuffle=True, random_state=42
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 2: Return the configured splitter instance
    # This object implements .split(X, y) and .get_n_splits()
    return splitter
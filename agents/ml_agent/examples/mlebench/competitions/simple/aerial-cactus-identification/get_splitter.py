import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Task-adaptive type definitions
X = pd.Series      # Feature matrix: Series of absolute image paths
y = pd.Series      # Target vector: Series of binary labels (int)

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    This function implements a Stratified 5-Fold Cross-Validation strategy to
    ensure that the class distribution (75/25) of the 'has_cactus' target
    is preserved across all validation folds. This is critical for reliable
    ROC AUC estimation.

    Args:
        X (X): The full training features (Series of image paths).
        y (y): The full training targets (Series of binary labels).

    Returns:
        StratifiedKFold: A splitter object configured with 5 splits,
                         shuffling, and a fixed random state for reproducibility.
    """
    # Step 1: Analyze task type and data characteristics
    # The task is a binary classification problem with imbalanced classes.
    # Stratified splitting is required to maintain the evaluation metric's stability.

    # Step 2: Select appropriate splitter based on analysis
    # Defined per technical specification: n_splits=5, shuffle=True, random_state=42.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 3: Return configured splitter instance
    # The returned object implements .split(X, y) and .get_n_splits().
    return splitter
import pandas as pd
from typing import Any
from sklearn.model_selection import StratifiedKFold, KFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/herbarium-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/cf84c6d3-8647-45ca-b9ff-02e7ed67cf5b/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing file paths and metadata
y = pd.Series         # Main target vector (category_id)

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.
    
    Uses StratifiedKFold to maintain species representation across folds, 
    essential for accurate Macro F1 estimation in this long-tailed dataset.
    
    Includes a robustness check: if the data subset (e.g., in validation_mode) 
    contains classes with fewer samples than the requested number of splits, 
    it falls back to KFold to prevent a ValueError while maintaining 
    shuffling and reproducibility.

    Args:
        X (pd.DataFrame): The full training features.
        y (pd.Series): The full training targets (category_id).

    Returns:
        Any: A splitter object (StratifiedKFold or KFold) configured for 3-fold validation.
    """
    n_splits = 3
    random_state = 42
    shuffle = True

    # Step 1: Analyze data characteristics for stratification feasibility
    # StratifiedKFold requires at least n_splits samples per class.
    # The full dataset has min 3 samples, but subsets (validation_mode) may have 1 or 2.
    min_class_count = y.value_counts().min()

    # Step 2: Select appropriate splitter based on feasibility
    if min_class_count < n_splits:
        # Fallback to KFold to ensure pipeline execution on small data subsets
        # while keeping the same split count and randomness.
        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    else:
        # Primary strategy: Stratified K-Fold as per Technical Specification
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    # Step 3: Return configured splitter instance
    return splitter
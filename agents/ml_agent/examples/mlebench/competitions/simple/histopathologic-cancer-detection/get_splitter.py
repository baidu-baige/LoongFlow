import pandas as pd
from typing import Any
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/cf83edc4-8764-4cf8-95a0-4f4a823260c7/2/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix type: DataFrame with columns ['id', 'path']
y = pd.Series         # Target vector type: Series of binary labels

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    This function implements a Stratified K-Fold cross-validation strategy 
    to ensure class balance (60/40) is preserved across folds and every 
    training sample is used for validation once.

    Args:
        X (X): The full training features (DataFrame with 'id' and 'path').
        y (y): The full training targets (Series of binary labels).

    Returns:
        StratifiedKFold: A configured splitter object for 5-fold cross-validation.
    """
    print("Defining Stratified K-Fold splitting strategy...")
    
    # StratifiedKFold ensures that the ratio of positive/negative labels 
    # in each fold is consistent with the whole dataset.
    # n_splits=5 provides a 80/20 train/val split per fold.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Note: X and y are validated to be aligned in the load_data stage.
    # The splitter.split(X, y) method will be called in the downstream workflow.
    
    print(f"Splitter initialized: StratifiedKFold(n_splits={splitter.get_n_splits()}, shuffle=True, random_state=42)")
    
    return splitter
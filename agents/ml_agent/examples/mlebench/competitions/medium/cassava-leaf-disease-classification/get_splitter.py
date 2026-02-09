from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/502e395f-5bca-4b30-9a81-fc7b49a1e544/3/executor/output"

# Task-adaptive type definitions based on upstream load_data component
X = pd.DataFrame  # Feature matrix: DataFrame containing 'image_path'
y = pd.Series     # Target vector: Series containing 'label'

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    For the Cassava Leaf Disease Classification task, a Stratified K-Fold strategy is 
    employed to handle the significant class imbalance (e.g., Class 3 is dominant). 
    This ensures each fold maintains a representative distribution of the five 
    disease categories.

    Args:
        X (X): The full training features (DataFrame containing 'image_path').
        y (y): The full training targets (Series containing 'label').

    Returns:
        StratifiedKFold: A splitter object that implements:
            - split(X, y, groups=None) -> Iterator[(train_idx, val_idx)]
            - get_n_splits() -> int
    """
    # Step 1: Initialize StratifiedKFold
    # n_splits=5: Standard 5-fold cross-validation
    # shuffle=True: Randomizes the data before splitting
    # random_state=42: Ensures reproducibility across different runs
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 2: Return configured splitter instance
    return splitter
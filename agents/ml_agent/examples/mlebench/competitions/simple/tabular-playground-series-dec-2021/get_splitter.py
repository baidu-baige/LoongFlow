from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/477e9955-ebee-46f4-96b0-878df6f022f5/1/executor/output"

# Task-adaptive type definitions aligned with upstream load_data
X = pd.DataFrame      # Feature matrix type
y = pd.Series         # Target vector type

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a StratifiedKFold splitting strategy for model validation.
    
    This strategy is chosen to maintain consistent class ratios across folds,
    which is critical given the extreme class imbalance in the Cover_Type target
    (specifically for minority classes like Class 4).

    Args:
        X (pd.DataFrame): The training feature set.
        y (pd.Series): The training target labels.

    Returns:
        StratifiedKFold: A configured cross-validator object.
    """
    # Technical Specification:
    # Method: StratifiedKFold
    # Parameters: n_splits=5, shuffle=True, random_state=42
    
    splitter = StratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    return splitter
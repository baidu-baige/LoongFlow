import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/seti-breakthrough-listen/prepared/public"
OUTPUT_DATA_PATH = "output/a429c40e-fe12-455c-8b05-ca9d732aabeb/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix containing 'id' and 'filepath'
y = pd.Series     # Target vector (binary: 0 or 1)

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    For this imbalanced SETI signal detection task (9.9% positive rate), 
    StratifiedKFold is used to ensure each fold maintains the global target 
    distribution, preventing validation bias.

    Args:
        X (X): The full training features.
        y (y): The full training targets.

    Returns:
        StratifiedKFold: A splitter object with split() and get_n_splits() methods.
    """
    # Technical Specification:
    # - Method: StratifiedKFold
    # - Parameters: n_splits=5, shuffle=True, random_state=42
    # - Objective: Reliable performance estimation on imbalanced data
    
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    return splitter
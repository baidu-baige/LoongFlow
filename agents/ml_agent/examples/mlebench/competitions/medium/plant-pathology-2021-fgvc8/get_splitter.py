from typing import Any
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/e81e4df9-fbfb-4465-8b24-4be8ee1f51f4/1/executor/output"

# Concrete types based on upstream load_data component
X = pd.Series    # Series of full image paths
y = np.ndarray  # Binary matrix (N, 6)

def get_splitter(X: X, y: y) -> MultilabelStratifiedKFold:
    """
    Defines and returns a MultilabelStratifiedKFold splitting strategy for model validation.

    This ensures that each fold is representative of the multi-label distribution, 
    maintaining the proportions of the 6 disease categories across training and 
    validation sets.

    Args:
        X (X): The full training features (Series of image paths).
        y (y): The full training targets (Binary matrix of shape [N, 6]).

    Returns:
        MultilabelStratifiedKFold: A splitter object that implements:
            - split(X, y) -> Iterator[(train_idx, val_idx)]
            - get_n_splits() -> int
    """
    # MultilabelStratifiedKFold is required because images can have multiple labels 
    # and simple KFold or StratifiedKFold (single-label) would not preserve the 
    # label distribution effectively.
    
    n_splits = 5
    shuffle = True
    random_state = 42

    print(f"Defining validation strategy: MultilabelStratifiedKFold(n_splits={n_splits}, shuffle={shuffle}, random_state={random_state})")

    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    return splitter
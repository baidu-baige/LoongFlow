from typing import Any
import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/6795de84-a7ab-443d-bbf5-03771db15966/1/executor/output"

# Task-adaptive type definitions
X = np.ndarray      # Feature matrix: [N, 4000] float32
y = np.ndarray      # Target vector: [N] int64

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a Stratified K-Fold data splitting strategy for model validation.
    
    This strategy ensures that the class distribution (whale vs noise) is preserved 
    across each of the folds, which is critical for the imbalanced nature of this 
    bioacoustics classification task.

    Args:
        X (X): The full training features.
        y (y): The full training targets.

    Returns:
        StratifiedKFold: A splitter object configured for 5-fold stratified validation.
    """
    # Technical Specification implementation:
    # - Method: Stratified K-Fold
    # - n_splits: 5
    # - shuffle: True
    # - random_state: 42 (ensures reproducibility)
    
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    return splitter
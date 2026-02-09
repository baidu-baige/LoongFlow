import pandas as pd
import numpy as np
from typing import Dict, Union, Iterator, Tuple, Any
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/bdc750a4-f0a3-4926-871d-f9675d7cf1ef/1/executor/output"

# Task-adaptive type definitions
X = Dict[Union[int, str], pd.DataFrame]  # Dictionary mapping segment_id to seismic signal DataFrame
y = pd.Series                            # Target time_to_eruption indexed by segment_id

class StratifiedQuantileKFold:
    """
    A validation splitter that implements Stratified K-Fold for continuous targets
    by partitioning the target variable into discrete quantiles.
    """
    def __init__(self, n_splits: int = 5, n_bins: int = 10, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: X, y: y, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets.
        
        Args:
            X (X): The full training features (dictionary of DataFrames).
            y (y): The full training targets (Series of time_to_eruption).
            groups (Any): Group labels for the samples (not used).
            
        Returns:
            Iterator: A generator yielding (train_idx, val_idx) tuples.
        """
        # Stratify based on 10 quantiles of the target 'time_to_eruption'
        # This ensures each fold contains a representative range of eruption times.
        # duplicates='drop' is used to handle cases with heavy-tailed distributions or low variance.
        y_bins = pd.qcut(y, q=self.n_bins, labels=False, duplicates='drop')
        
        # We use a simple integer array to generate indices from StratifiedKFold
        indices = np.arange(len(y))
        
        # Delegate splitting to the underlying StratifiedKFold instance
        yield from self.skf.split(indices, y_bins)

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

def get_splitter(X: X, y: y) -> StratifiedQuantileKFold:
    """
    Defines and returns the data partitioning strategy for cross-validation.

    The strategy uses Stratified K-Fold on 10 quantiles of the 'time_to_eruption' 
    target to ensure that the distribution of targets is balanced across folds, 
    which is critical for the stability of the Mean Absolute Error (MAE) metric.

    Args:
        X (X): The full training features.
        y (y): The full training targets.

    Returns:
        StratifiedQuantileKFold: A splitter instance configured with 5-fold stratification.
    """
    # Parameters defined per Technical Specification:
    # Method: Stratified K-Fold
    # Target: 10 quantiles of time_to_eruption
    # Splits: 5
    # Shuffle: True
    # Random State: 42
    return StratifiedQuantileKFold(
        n_splits=5, 
        n_bins=10, 
        shuffle=True, 
        random_state=42
    )
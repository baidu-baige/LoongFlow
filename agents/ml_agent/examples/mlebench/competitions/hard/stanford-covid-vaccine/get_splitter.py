import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Any, Iterator, Tuple, Union

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.DataFrame

class ShuffledGroupKFold:
    """
    A K-Fold cross-validator that provides train/test indices to split data based on groups,
    while also supporting shuffling and a random seed.
    
    This implementation ensures that all observations belonging to the same group 
    (e.g., all bases of a single RNA molecule identified by 'id') are assigned 
    to the same fold, preventing sequence-level leakage.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and validation sets.
        
        Args:
            X: The feature matrix.
            y: The target matrix.
            groups: Group identifiers for the samples used while splitting the dataset into 
                   train/test set. If None, it attempts to use 'id' column from X.
        
        Yields:
            train_idx: The training set indices for that split.
            val_idx: The validation set indices for that split.
        """
        if groups is None:
            if isinstance(X, pd.DataFrame) and 'id' in X.columns:
                groups = X['id'].values
            else:
                # If no groups are provided or found, fall back to standard KFold on indices
                kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
                yield from kf.split(X, y)
                return

        groups = np.array(groups)
        unique_groups = np.unique(groups)
        
        # Split the unique groups to ensure group-level isolation
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        for train_group_indices, val_group_indices in kf.split(unique_groups):
            train_groups = unique_groups[train_group_indices]
            val_groups = unique_groups[val_group_indices]
            
            # Map group-level splits back to the original indices in the data
            train_idx = np.where(np.isin(groups, train_groups))[0]
            val_idx = np.where(np.isin(groups, val_groups))[0]
            
            yield train_idx, val_idx

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    The strategy implements a GroupKFold approach that supports shuffling, ensuring that 
    all data points (bases) related to a specific RNA molecule (group) stay together 
    in either training or validation to prevent information leakage.

    Args:
        X (pd.DataFrame): The training features containing at least the 'id' column.
        y (pd.DataFrame): The training targets.

    Returns:
        ShuffledGroupKFold: A splitter object that partitions data based on the 'id' column.
    """
    
    # Initialize the custom splitter with specified parameters from technical specification
    splitter = ShuffledGroupKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    print(f"Initialized ShuffledGroupKFold with {splitter.get_n_splits()} splits.")
    
    return splitter
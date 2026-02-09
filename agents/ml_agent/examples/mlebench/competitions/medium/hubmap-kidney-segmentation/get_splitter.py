import pandas as pd
import numpy as np
from typing import Any, Iterator, Tuple
from sklearn.model_selection import KFold

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix containing metadata and image paths
y = pd.Series     # Target vector containing RLE strings

class ShuffledGroupKFold:
    """
    A custom splitter that implements GroupKFold logic with shuffling support.
    Standard sklearn.model_selection.GroupKFold does not support shuffle or random_state.
    This implementation partitions data such that the same group (patient) is never 
    in both training and validation sets, while allowing for reproducible shuffling.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        # Use KFold internally to shuffle the unique groups themselves
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and validation sets.
        """
        if groups is None:
            # According to technical spec, use 'patient_number' as the grouping variable
            if isinstance(X, pd.DataFrame) and 'patient_number' in X.columns:
                groups = X['patient_number'].values
            else:
                raise KeyError("The grouping variable 'patient_number' was not found in X, and no groups were provided.")
        
        # Ensure groups is a numpy array for indexing
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        
        if len(unique_groups) < self.n_splits:
            raise ValueError(f"Number of groups ({len(unique_groups)}) is less than n_splits ({self.n_splits}).")

        # Split the unique groups
        for train_group_idx, val_group_idx in self.kf.split(unique_groups):
            train_groups = unique_groups[train_group_idx]
            val_groups = unique_groups[val_group_idx]
            
            # Map back from unique groups to the indices of the original data
            train_idx = np.where(np.isin(groups, train_groups))[0]
            val_idx = np.where(np.isin(groups, val_groups))[0]
            
            yield train_idx, val_idx

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.
    
    The strategy uses GroupKFold logic partitioned by 'patient_number' to ensure 
    the model generalizes to unseen patients, preventing data leakage.
    
    Args:
        X (X): The training features (metadata DataFrame).
        y (y): The training targets (RLE strings).

    Returns:
        Any: A ShuffledGroupKFold splitter object.
    """
    print("Stage 2: Defining data partitioning strategy...")
    
    # Initialize the splitter with parameters from the technical specification
    splitter = ShuffledGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    print(f"Splitter defined: GroupKFold (shuffled) with n_splits=5 on 'patient_number'")
    
    return splitter
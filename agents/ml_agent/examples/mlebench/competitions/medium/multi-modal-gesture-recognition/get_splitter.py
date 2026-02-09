import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Any, Iterator, Tuple

# Task-adaptive type definitions
# X: List of dicts, each containing 'id', 'skeleton', 'audio', 'fs', 'precise_labels'
# y: List of lists, each containing the ground truth gesture sequence (integers 1-20)
X = List[Dict[str, Any]]
y = List[List[int]]

class ShuffledGroupKFold:
    """
    Implementation of GroupKFold that supports shuffling and random_state.
    This ensures the validation strategy mimics the user-independent requirement
    of the competition by splitting data at the session level, preventing
    information leakage if multiple segments belong to the same session/user.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and validation set.

        Args:
            X (X): The full training features (List of Dicts).
            y (y, optional): The full training targets.
            groups (Any, optional): Group labels for the samples. If None, extracted from 'id' in X.

        Yields:
            train_idx (np.ndarray): The training set indices for that split.
            val_idx (np.ndarray): The validation set indices for that split.
        """
        # Extract Session/Sample IDs from features if groups are not explicitly provided.
        if groups is None:
            # Each entry in X is a dictionary containing 'id' (the 4-digit SampleID string)
            # This ID represents a specific recording session.
            groups = np.array([sample['id'] for sample in X])
        else:
            groups = np.array(groups)
        
        # Identify unique groups
        unique_groups = np.unique(groups)
        
        # Check if enough groups exist for the requested splits
        if len(unique_groups) < self.n_splits:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater"
                f" than the number of groups: {len(unique_groups)}"
            )
        
        # Use KFold on unique group IDs to achieve shuffled group-wise splitting.
        # This is necessary because sklearn.model_selection.GroupKFold does not support shuffle.
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        indices = np.arange(len(groups))
        
        for train_group_idx, val_group_idx in kf.split(unique_groups):
            # Select the group IDs for this fold
            train_groups = unique_groups[train_group_idx]
            val_groups = unique_groups[val_group_idx]
            
            # Create masks to map group-level splits back to sample-level indices
            # np.isin is vectorized and memory-efficient for this mapping
            train_mask = np.isin(groups, train_groups)
            val_mask = np.isin(groups, val_groups)
            
            yield indices[train_mask], indices[val_mask]

def get_splitter(X: X, y: y) -> ShuffledGroupKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    The strategy uses session-independent logic (ShuffledGroupKFold) to provide 
    a robust estimate of performance on unseen users/sessions, directly 
    addressing the competition requirement for user-independent learning.

    Args:
        X (X): The full training features.
        y (y): The full training targets.

    Returns:
        ShuffledGroupKFold: A splitter object that implements split() and get_n_splits().
    """
    # Initialize the splitter with 5-fold cross-validation.
    # Shuffling is enabled to prevent temporal or ordering bias while 
    # maintaining session/user integrity for validation.
    splitter = ShuffledGroupKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    return splitter
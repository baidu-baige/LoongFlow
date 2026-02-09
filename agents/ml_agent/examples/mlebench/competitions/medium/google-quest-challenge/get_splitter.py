import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Any, Iterator, Tuple, Optional

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing 'question_title' for grouping
y = pd.DataFrame      # Target vector with 30 continuous labels

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/aaa741b3-cb02-44fc-a666-dd434e563444/8/executor/output"

class GroupKFold:
    """
    A GroupKFold implementation that supports shuffling and random_state.
    
    Standard sklearn.model_selection.GroupKFold does not support shuffle and random_state 
    parameters. This implementation partitions unique group IDs (e.g., question_title) 
    using a shuffled KFold, ensuring that all samples with the same group ID stay 
    together in either the training or validation set, preventing data leakage from 
    duplicated questions.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Optional[X] = None, y: Optional[y] = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

    def split(self, X: X, y: Optional[y] = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and validation sets.
        
        Args:
            X: Training features.
            y: Training targets (optional).
            groups: Group labels for the samples. If None, defaults to 'question_title' in X.
            
        Yields:
            train_idx: The training set indices for that split.
            val_idx: The validation set indices for that split.
        """
        # Determine the grouping array
        if groups is None:
            if isinstance(X, pd.DataFrame) and 'question_title' in X.columns:
                groups = X['question_title']
            else:
                raise ValueError("The 'groups' parameter must be provided or 'question_title' must exist in X.")
        
        # Ensure groups is a numpy array for consistent indexing
        groups_arr = np.asarray(groups)
        # return_inverse provides an integer mapping for each row to its unique group
        unique_groups, group_indices = np.unique(groups_arr, return_inverse=True)
        
        # Partition the unique group identifiers using KFold to allow shuffling and reproducibility
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        for train_grp_idx, val_grp_idx in kf.split(unique_groups):
            # Map group-level split back to original sample indices
            # group_indices contains the index of each sample's group in unique_groups
            train_mask = np.isin(group_indices, train_grp_idx)
            val_mask = np.isin(group_indices, val_grp_idx)
            
            yield np.where(train_mask)[0], np.where(val_mask)[0]

def get_splitter(X: X, y: y) -> GroupKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    This function utilizes a GroupKFold strategy based on 'question_title' to 
    prevent the same question from appearing in both training and validation sets.
    This handles the data leakage risk posed by the ~2,000 duplicate questions 
    in the training set, ensuring the validation performance generalizes to the 
    unique questions in the test set.

    Args:
        X (X): The full training features.
        y (y): The full training targets.

    Returns:
        GroupKFold: A splitter object that implements 5-fold cross-validation 
                    with group-level shuffling for reproducibility.
    """
    print("Execution Stage: get_splitter")

    # Initialize the custom GroupKFold splitter.
    # We use 5 folds as a standard balance between validation reliability and compute time.
    # Grouping on 'question_title' is the optimal strategy as per technical specification.
    splitter = GroupKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )

    print(f"Splitter defined: GroupKFold(n_splits=5, shuffle=True, random_state=42)")
    print("Validation Strategy: Grouped by 'question_title' to ensure zero leakage from duplicate questions.")
    
    return splitter
from typing import Any, Iterator, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/02d42284-9bf3-4f97-ab6c-7ea839095b54/3/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix containing 'anchor', 'target', 'context', and 'context_desc'
y = pd.Series     # Target vector containing the 'score'

class GroupKFoldShuffled:
    """
    A custom splitter that implements GroupKFold with shuffle and random_state support.
    
    Standard scikit-learn GroupKFold does not support shuffling or a random seed.
    This implementation partitions the unique groups using a shuffled KFold, 
    ensuring that all rows with the same group ID (anchor) stay together 
    while providing reproducibility and variability across folds.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets.
        
        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series, optional): The target vector.
            groups (Any, optional): Group labels. If None, 'anchor' column from X is used.
            
        Yields:
            train_idx (np.ndarray): The training set indices for that split.
            val_idx (np.ndarray): The validation set indices for that split.
        """
        # Technical Specification: Partition data such that 'anchor' occurrences are non-overlapping
        # This is critical for the Patent Phrase Matching task to ensure the model generalizes 
        # to unseen anchor phrases.
        if groups is None:
            if 'anchor' not in X.columns:
                raise KeyError("The 'anchor' column is required for grouping but was not found in X.")
            groups = X['anchor']
        
        # Ensure groups is a pandas Series for easy processing and alignment
        groups_series = pd.Series(groups).reset_index(drop=True)
        unique_groups = groups_series.unique()
        
        # Use KFold on the unique group IDs (anchors) to determine which groups go into which fold.
        # This achieves both group-wise splitting and shuffling with a fixed seed.
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        indices = np.arange(len(X))
        
        for train_group_indices, val_group_indices in kf.split(unique_groups):
            train_groups = unique_groups[train_group_indices]
            val_groups = unique_groups[val_group_indices]
            
            # Map the selected groups back to the original index positions
            # We use isin() for efficient lookup across the group series
            train_mask = groups_series.isin(train_groups).values
            val_mask = groups_series.isin(val_groups).values
            
            yield indices[train_mask], indices[val_mask]

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    This function determines HOW to partition data for training vs validation.
    For the US Patent Phrase Matching task, we use a GroupKFold strategy 
    grouped by the 'anchor' column. This prevents information leakage where 
    the same anchor phrase appears in both training and validation sets, 
    simulating the challenge of matching phrases for novel anchors.

    Args:
        X (X): The full training features (pd.DataFrame).
        y (y): The full training targets (pd.Series).

    Returns:
        Any: A GroupKFoldShuffled splitter instance configured with 5 folds.
    """
    print("Execution: get_splitter")
    
    # Validation check: Ensure 'anchor' exists in X to support grouping logic
    if 'anchor' not in X.columns:
        raise ValueError("Feature matrix X must contain 'anchor' column for GroupKFold splitting.")

    # Step 1: Initialize the custom GroupKFoldShuffled splitter.
    # We use a 5-fold strategy which is standard for datasets of this size (~33k rows).
    # Shuffle is enabled with a fixed random state (42) for reproducibility.
    splitter = GroupKFoldShuffled(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    print(f"Initialized GroupKFoldShuffled splitter with {splitter.n_splits} splits (grouping by 'anchor').")
    
    return splitter
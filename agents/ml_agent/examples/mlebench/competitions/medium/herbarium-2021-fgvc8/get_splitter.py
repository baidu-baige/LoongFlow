import numpy as np
import pandas as pd
from typing import Any, Iterator, Tuple

# Task-adaptive type definitions
# X is a DataFrame containing 'image_id' and 'path'
# y is a DataFrame containing 'category_id', 'family_id', and 'order_id'
X = pd.DataFrame
y = pd.DataFrame

class HerbariumSplitter:
    """
    A validation splitter implementation designed for the Herbarium 2021 dataset.
    This splitter implements a robust stratification strategy that handles the 
    extreme long-tailed distribution (64,500 classes, some with < 5 samples) 
    without raising the membership count errors common in standard StratifiedKFold.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: X, y: y, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets while
        ensuring every category_id is represented across folds.
        
        Args:
            X (pd.DataFrame): The full training features.
            y (pd.DataFrame): The training targets containing 'category_id'.
            groups: Placeholder for API compatibility.
            
        Returns:
            Iterator of (train_idx, val_idx) as numpy arrays.
        """
        # Extract category labels for stratification
        y_cat = y['category_id'].values
        n_samples = len(y_cat)
        
        # We group indices by category_id using a stable sort (O(N log N)).
        # This is memory-efficient and fast for 1.7M records on the available 36-core CPU.
        sort_idx = np.argsort(y_cat, kind='stable')
        sorted_y = y_cat[sort_idx]
        
        # Identify boundaries where category_id changes to isolate each class
        diff = np.diff(sorted_y)
        boundaries = np.concatenate(([0], np.where(diff != 0)[0] + 1, [n_samples]))
        
        fold_assignments = np.zeros(n_samples, dtype=int)
        rng = np.random.default_rng(self.random_state)
        
        # Distribute samples of each class across the folds
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            class_indices = sort_idx[start:end]
            
            if self.shuffle:
                rng.shuffle(class_indices)
            
            count = len(class_indices)
            # Use a random offset for the starting fold of each class.
            # This ensures that rare classes (e.g., 1 sample) are distributed 
            # across all folds' validation sets over different iterations, 
            # rather than all being concentrated in the first fold.
            offset = rng.integers(self.n_splits)
            fold_assignments[class_indices] = (np.arange(count) + offset) % self.n_splits
            
        indices = np.arange(n_samples)
        for fold in range(self.n_splits):
            train_mask = fold_assignments != fold
            val_mask = fold_assignments == fold
            # Yielding indices as required by sklearn-style splitter API
            yield indices[train_mask], indices[val_mask]

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a robust data partitioning strategy for model validation.
    
    This function creates a custom splitter that preserves the category distribution 
    (stratification) while gracefully handling classes with fewer samples than 
    the number of splits.

    Args:
        X (pd.DataFrame): The full training features.
        y (pd.DataFrame): The full training targets (including category_id).

    Returns:
        HerbariumSplitter: A splitter object with split() and get_n_splits() methods.
    """
    return HerbariumSplitter(n_splits=5, shuffle=True, random_state=42)
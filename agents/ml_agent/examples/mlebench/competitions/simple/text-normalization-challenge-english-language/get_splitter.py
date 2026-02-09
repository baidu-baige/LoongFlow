import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from typing import Any, Iterator, Tuple

# Task-adaptive type definitions for English Text Normalization
X = pd.DataFrame
y = pd.Series

class GroupKFold:
    """
    A scikit-learn compatible splitter that performs K-Fold cross-validation 
    while ensuring that all tokens from the same sentence (sentence_id) 
    are kept together in the same fold. This implementation supports 
    shuffling, as required by the technical specification.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets.
        
        Ensures that all samples sharing the same group ID appear together 
        in either the training or the validation set.
        """
        # If groups are not provided as an argument, extract 'sentence_id' from features
        if groups is None:
            if isinstance(X, pd.DataFrame) and 'sentence_id' in X.columns:
                groups = X['sentence_id'].values
            else:
                raise ValueError("The 'sentence_id' column must be present in X or 'groups' must be provided to define sentence boundaries.")
        
        # Efficiently identify unique group IDs and their mapping to the original array
        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        
        # Split the unique sentence IDs using standard KFold to respect shuffle/random_state
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        # Pre-allocate fold assignments for each unique group (memory-efficient int8)
        group_to_fold = np.empty(len(unique_groups), dtype=np.int8)
        for fold_idx, (_, val_group_indices) in enumerate(kf.split(unique_groups)):
            group_to_fold[val_group_indices] = fold_idx
            
        # Broadcast the fold assignments from unique groups back to individual tokens
        sample_to_fold = group_to_fold[group_indices]
        
        # Yield indices for each fold
        for fold_idx in range(self.n_splits):
            train_idx = np.where(sample_to_fold != fold_idx)[0]
            val_idx = np.where(sample_to_fold == fold_idx)[0]
            yield train_idx, val_idx

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    Strategy: GroupKFold based on sentence_id to prevent data leakage between
    training and validation sets, ensuring that the model generalizes to 
    new linguistic contexts and unseen sentences.
    """
    # Initialize the splitter with parameters defined in the technical specification
    splitter = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    return splitter
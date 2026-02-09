import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Any
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Concrete type definitions for this task
X = pd.DataFrame  # Metadata including 'is_curated' flag
y = pd.DataFrame  # Multi-hot encoded labels (80 columns)

class AudioTaskSplitter:
    """
    Implements a cross-validation strategy tailored for the Freesound Audio Tagging 2019 task.
    
    The strategy performs MultilabelStratifiedKFold on the curated (clean) data only, 
    ensuring validation is conducted on a reliable subset. For training, it combines 
    the training portion of the curated data with the entire noisy dataset to 
    maximize the supervision available while preventing noisy data leakage into validation.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        # Initialize the underlying multi-label stratifier
        self.mskf = MultilabelStratifiedKFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )

    def split(self, X: X, y: y, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for training and validation splits.
        """
        # Identify global indices for curated and noisy subsets
        curated_mask = X['is_curated'].values
        curated_indices = np.where(curated_mask)[0]
        noisy_indices = np.where(~curated_mask)[0]

        # Extract targets for the curated subset for stratification purposes
        y_curated = y.iloc[curated_indices].values
        
        # Split only the curated indices
        # We pass a range for X to mskf as it primarily uses y for stratification logic
        for train_rel_idx, val_rel_idx in self.mskf.split(np.arange(len(curated_indices)), y_curated):
            # Map relative indices (0 to len(curated)-1) back to global indices in X
            global_cur_train_idx = curated_indices[train_rel_idx]
            global_cur_val_idx = curated_indices[val_rel_idx]

            # Per Technical Specification:
            # Training set = (curated_train_subset + all_noisy_data)
            # Validation set = (curated_val_subset)
            train_idx = np.concatenate([global_cur_train_idx, noisy_indices])
            val_idx = global_cur_val_idx

            # To ensure the training loop doesn't see data in a rigid block (curated then noisy),
            # the training loop should shuffle these indices. We return them as defined.
            yield train_idx, val_idx

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns the 5-fold MultilabelStratifiedKFold splitting strategy.
    
    Args:
        X (X): The full training features (metadata DataFrame).
        y (y): The full training targets (multi-hot DataFrame).

    Returns:
        AudioTaskSplitter: A splitter object that partitions curated data while augmenting training folds with noisy data.
    """
    # Parameters as per Technical Specification
    return AudioTaskSplitter(n_splits=5, shuffle=True, random_state=42)
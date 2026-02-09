import cudf
import numpy as np
from sklearn.model_selection import KFold
from typing import Any, Iterator, Tuple

# Task-adaptive type definitions for RAPIDS cuDF
# Utilizing cuDF for high-performance GPU data processing of the ~10M tokens.
X = cudf.DataFrame
y = cudf.Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/ecfe1a48-59fb-4170-a38b-6ffb4a298ec0/10/executor/output"

class GroupKFoldShuffled:
    """
    Implements a GroupKFold strategy that supports shuffling and random_state.
    
    Standard scikit-learn GroupKFold implementations are deterministic and 
    do not support shuffling. This implementation partitions the unique group IDs 
    (sentence_id) using a shuffled KFold and maps those partitions back to the 
    original dataset indices using GPU-accelerated operations.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets based on groups.
        
        Args:
            X: The feature matrix (cuDF DataFrame).
            y: The target vector (cuDF Series).
            groups: Group labels for the samples. Defaults to 'sentence_id' from X.
        
        Yields:
            train_idx: The training set indices for that split.
            val_idx: The validation set indices for that split.
        """
        if groups is None:
            # Default to 'sentence_id' which is the natural grouping for this task.
            # Tokens from the same sentence must stay together to prevent leakage.
            if hasattr(X, 'columns') and 'sentence_id' in X.columns:
                groups = X['sentence_id']
            else:
                raise ValueError("The 'groups' parameter or 'sentence_id' column in X is required.")

        # Ensure 'groups' is a cuDF Series for efficient GPU processing
        if not isinstance(groups, cudf.Series):
            groups = cudf.Series(groups)

        # Step 1: Extract unique group IDs (sentences) and move to CPU for KFold logic
        # The number of unique sentences (~685k) is small enough for CPU memory and KFold processing.
        unique_groups = groups.unique().to_numpy()
        
        # Step 2: Define KFold on the unique group IDs to achieve shuffled group splitting
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        # Pre-allocate full index range on CPU
        all_indices = np.arange(len(groups))
        
        for _, val_group_idx in kf.split(unique_groups):
            # Identify which group values belong to the validation set
            val_group_vals = unique_groups[val_group_idx]
            
            # Step 3: Use GPU-accelerated 'isin' to create a mask for the full dataset.
            # Converting val_group_vals to cuDF Series ensures the operation stays on GPU.
            # This is critical for performance given the 10M row scale.
            val_group_vals_gpu = cudf.Series(val_group_vals)
            val_mask_gpu = groups.isin(val_group_vals_gpu)
            
            # Move boolean mask to CPU to perform indexing
            val_mask = val_mask_gpu.to_numpy()
            
            # Yield train and validation indices as numpy arrays
            yield all_indices[~val_mask], all_indices[val_mask]

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.
    
    This implementation uses a custom Shuffled GroupKFold on 'sentence_id' to:
    1. Prevent data leakage (tokens from the same sentence are never split between train/val).
    2. Ensure reproducible, shuffled validation folds.
    3. Maximize performance using GPU-accelerated group membership checks.

    Args:
        X (X): The full training features (cuDF DataFrame).
        y (y): The full training targets (cuDF Series).

    Returns:
        GroupKFoldShuffled: A splitter object with split() and get_n_splits() methods.
    """
    # Use 5 folds to balance compute time and validation reliability.
    # Grouping by sentence_id is strictly required per task logic to avoid leakage.
    return GroupKFoldShuffled(n_splits=5, shuffle=True, random_state=42)
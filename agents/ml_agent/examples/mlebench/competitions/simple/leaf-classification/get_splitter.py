import cudf
import numpy as np
from typing import Any, Iterator, Tuple
from cuml.model_selection import StratifiedKFold
from sklearn.model_selection import KFold as SklearnKFold

# Task-adaptive type definitions using RAPIDS for GPU-accelerated processing
X = cudf.DataFrame      # Feature matrix type: RAPIDS DataFrame
y = cudf.Series         # Target vector type: RAPIDS Series (encoded species labels)

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/leaf-classification/prepared/public"
OUTPUT_DATA_PATH = "output/5e63fe40-52af-4d8b-ac71-4d3a91b9999f/54/executor/output"

class GPUCompatibleKFold:
    """
    A wrapper around sklearn's KFold to support GPU-backed cudf objects.
    Ensures .split() works seamlessly with cudf types by yielding indices based on length.
    """
    def __init__(self, n_splits: int, shuffle: bool, random_state: int):
        self.n_splits = n_splits
        self.splitter = SklearnKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets.
        Uses numpy-based index generation to maintain compatibility with cudf objects downstream.
        """
        indices = np.arange(len(X))
        return self.splitter.split(indices)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a 9-fold data splitting strategy for model validation.

    This implementation defaults to StratifiedKFold to maintain the class distribution 
    (99 species) across folds. If the minimum samples per class is less than 9 
    (common during validation_mode subsetting), it falls back to a standard 
    K-Fold strategy to ensure the pipeline remains executable.

    Args:
        X (X): The full training features (cudf.DataFrame).
        y (y): The full training targets (cudf.Series).

    Returns:
        Any: A splitter object (cuml StratifiedKFold or GPUCompatibleKFold) 
             implementing split() and get_n_splits().
    """
    # Technical Specification: n_splits=9, shuffle=True, random_state=42.
    n_splits = 9
    random_state = 42
    
    # StratifiedKFold requires at least n_splits samples per class.
    # We check the distribution of the target labels.
    class_counts = y.value_counts()
    min_samples_per_class = int(class_counts.min())
    
    if min_samples_per_class >= n_splits:
        # Optimal case: Every class has enough samples for stratified 9-fold CV.
        # Use cuml's GPU-native implementation.
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
    else:
        # Fallback case: Dataset contains classes with fewer than 9 samples.
        # Standard K-Fold is used to maintain the 9-split structure.
        # cuml does not provide a standard KFold, so we use the sklearn wrapper.
        splitter = GPUCompatibleKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        
    return splitter
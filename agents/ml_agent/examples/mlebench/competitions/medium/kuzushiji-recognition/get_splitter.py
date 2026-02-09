import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Any, Iterator, Tuple

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series

class KuzushijiGroupSplitter:
    """
    A validation splitter that implements GroupKFold to ensure robust evaluation.
    By grouping by 'image_id', we ensure that all annotations/characters from 
    the same image are kept together in either the training or validation set,
    preventing information leakage and reflecting real-world inference scenarios.
    """
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.splitter = GroupKFold(n_splits=n_splits)

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[Any, Any]]:
        """
        Provides indices for training and validation sets.
        
        Args:
            X (pd.DataFrame): Training features containing 'image_id'.
            y (pd.Series): Training targets.
            groups (Any, optional): Grouping labels. Defaults to X['image_id'].
            
        Returns:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Train and validation indices.
        """
        # Technical Specification requires image_id as the grouping column
        if groups is None:
            # We assume X is the DataFrame returned by load_data containing 'image_id'
            # If 'image_id' is missing, we let the KeyError propagate to ensure visibility of data issues.
            groups = X['image_id']
            
        return self.splitter.split(X, y, groups=groups)

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    Args:
        X (X): The full training features (Detection DataFrame).
        y (y): The full training targets (Label strings).

    Returns:
        KuzushijiGroupSplitter: A splitter object implementing GroupKFold(n_splits=5).
    """
    # Implementation Details: 
    # - Method: GroupKFold
    # - Target: image_id as grouping column
    # - Parameters: n_splits = 5
    return KuzushijiGroupSplitter(n_splits=5)
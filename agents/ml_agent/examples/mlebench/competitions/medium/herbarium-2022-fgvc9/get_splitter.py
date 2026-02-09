import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Iterator, Tuple, Any

# Concrete types for this task: X is a DataFrame of file paths, y is a DataFrame of labels
X = pd.DataFrame
y = pd.DataFrame

class HerbariumSplitter:
    """
    Custom splitter that implements StratifiedKFold (n_splits=5) as required by technical specs,
    while providing robustness against the long-tail distribution where some classes 
    have fewer than 5 members (especially in validation/subsampled modes).
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: pd.DataFrame, y: pd.DataFrame = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for the first fold (Fold 0) of a 5-fold cross-validation.
        Uses stratification by category_id where possible.
        
        Args:
            X: Training features (DataFrame).
            y: Training targets (DataFrame containing 'category_id').
            
        Returns:
            Iterator yielding (train_idx, val_idx) for Fold 0.
        """
        if y is None or 'category_id' not in y.columns:
            # Fallback to standard KFold if labels are missing
            it = self.kf.split(X)
            yield next(it)
            return

        target = y['category_id']
        
        # Robust Stratification Logic:
        # StratifiedKFold requires at least `n_splits` members per class.
        # We group all classes with < n_splits members into a single 'rare' class bucket.
        counts = target.value_counts()
        rare_classes = counts[counts < self.n_splits].index
        
        if len(rare_classes) > 0:
            # Create a modified target for splitting purposes only
            # Classes with enough samples keep their ID; rare ones are grouped into -1
            strat_target = target.copy()
            strat_target[target.isin(rare_classes)] = -1
            
            # If after grouping, even the 'rare' bucket (or other classes) cannot satisfy n_splits,
            # we must check if we can still use StratifiedKFold on the new distribution.
            new_counts = strat_target.value_counts()
            if new_counts.min() < self.n_splits:
                # If even grouped stratification is impossible, fall back to KFold
                it = self.kf.split(X)
            else:
                it = self.skf.split(X, strat_target)
        else:
            # All classes have enough members
            it = self.skf.split(X, target)

        # Yield only the first fold (Fold 0) as per technical specification
        try:
            yield next(it)
        except StopIteration:
            raise RuntimeError("Splitter failed to generate any folds.")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """
        Returns 1 as the pipeline is configured for single-fold training (Fold 0) 
        to stay within the 24h time limit.
        """
        return 1

def get_splitter(X: X, y: y) -> HerbariumSplitter:
    """
    Defines the data partitioning strategy.
    
    This implementation adheres to the requirement for StratifiedKFold (n_splits=5) 
    stratified by category_id, while ensuring the code is resilient to the 
    subsampled data used in validation modes or extremely rare classes.
    """
    return HerbariumSplitter(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
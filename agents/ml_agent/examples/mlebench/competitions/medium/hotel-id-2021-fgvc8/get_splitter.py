import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from typing import Any, Iterator, Tuple

# Task-adaptive type definitions
X = pd.DataFrame  # Features: contains image_path, hotel_id, and group_id
y = pd.Series     # Target: encoded hotel_id

class HotelIdSplitter:
    """
    A robust validation strategy that implements StratifiedGroupKFold while handling 
    the high-cardinality and rare-class nature of the Hotel Recognition task.
    
    Key Features:
    1. Singleton Handling: Classes with only 1 sample are always assigned to the 
       training set to ensure the model sees every category at least once.
    2. Rare Class Grouping: Classes with fewer members than n_splits are grouped 
       into a proxy target to avoid the scikit-learn 'n_splits > n_samples' ValueError.
    3. Grouping: Uses 'group_id' (hotel_id + timestamp) to prevent leakage from image bursts.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: X, y: y, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # 1. Prepare and align indices
        y_ser = pd.Series(y).reset_index(drop=True)
        indices = np.arange(len(y_ser))
        
        if groups is None:
            if isinstance(X, pd.DataFrame) and 'group_id' in X.columns:
                groups_ser = X['group_id'].reset_index(drop=True)
            else:
                raise ValueError("Group identifiers ('group_id') must be provided for leakage-free splitting.")
        else:
            groups_ser = pd.Series(groups).reset_index(drop=True)

        # 2. Separate singletons as per Critical Constraints
        # Singletons (count == 1) MUST be in training for all folds.
        counts = y_ser.value_counts()
        singleton_mask = y_ser.isin(counts[counts == 1].index).values
        singleton_indices = indices[singleton_mask]
        other_indices = indices[~singleton_mask]
        
        # If there are no samples left to put into a validation set (unlikely in full data, 
        # possible in tiny validation subsets), yield a dummy split or propagate.
        if len(other_indices) < self.n_splits:
            # Fallback to GroupKFold on the whole set to ensure we at least get some folds
            splitter = GroupKFold(n_splits=self.n_splits)
            for t_idx, v_idx in splitter.split(X, y_ser, groups_ser):
                yield t_idx, v_idx
            return

        # 3. Handle data subset where count > 1
        X_other = X.iloc[other_indices].reset_index(drop=True)
        y_other = y_ser.iloc[other_indices].reset_index(drop=True)
        groups_other = groups_ser.iloc[other_indices].reset_index(drop=True)
        
        # 4. Prepare proxy target for StratifiedGroupKFold
        # Classes with [2, n_splits-1] members trigger the scikit-learn ValueError.
        # We group them into a single dummy class (-1) for stratification purposes.
        other_counts = y_other.value_counts()
        rare_classes = other_counts[other_counts < self.n_splits].index
        
        y_proxy = y_other.copy()
        if not rare_classes.empty:
            y_proxy[y_other.isin(rare_classes)] = -1
            
            # If the aggregate 'rare' class itself is still smaller than n_splits, 
            # merge it into the most frequent class to ensure the splitter doesn't crash.
            if 0 < (y_proxy == -1).sum() < self.n_splits:
                freq_counts = y_proxy[y_proxy != -1].value_counts()
                if not freq_counts.empty:
                    most_freq_label = freq_counts.idxmax()
                    y_proxy[y_proxy == -1] = most_freq_label

        # 5. Execute Splitting
        # Use StratifiedGroupKFold on proxy labels. If stratification is impossible 
        # (e.g., only 1 unique label in y_proxy), fallback to GroupKFold.
        if y_proxy.nunique() < 2:
            splitter = GroupKFold(n_splits=self.n_splits)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=self.n_splits, 
                shuffle=self.shuffle, 
                random_state=self.random_state
            )
        
        for train_idx_other, val_idx_other in splitter.split(X_other, y_proxy, groups_other):
            # Map back to original indices and combine with singletons
            # train_idx = [indices of other samples in train] + [indices of all singletons]
            train_idx = np.concatenate([other_indices[train_idx_other], singleton_indices])
            val_idx = other_indices[val_idx_other]
            yield train_idx, val_idx

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        return self.n_splits

def get_splitter(X: X, y: y) -> HotelIdSplitter:
    """
    Defines and returns the data splitting strategy.
    
    Args:
        X (X): Features (image paths, chain, group_id).
        y (y): Targets (encoded hotel_id).
        
    Returns:
        HotelIdSplitter: A splitter instance compatible with sklearn-style CV.
    """
    n_splits = 5
    shuffle = True
    random_state = 42
    
    print(f"Defining validation strategy: StratifiedGroupKFold (n_splits={n_splits}) with singleton protection and rare class proxying.")
    return HotelIdSplitter(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Any, Iterator, Tuple

# Task-adaptive type definitions for clear identification of input roles
X = Any  # Represents the WhaleDataset (torch.utils.data.Dataset)
y = pd.Series  # Represents the whale IDs (pd.Series)

class BinnedStratifiedKFold:
    """
    Implements a Stratified K-Fold strategy with Class Binning for rare classes.
    
    This splitter handles extreme class imbalance by grouping IDs with fewer samples 
     than 'n_splits' into a single pseudo-class ('rare_class_bin') for stratification logic.
    This ensures the StratifiedKFold algorithm remains stable and maintains representative 
    distributions across folds for classes that would otherwise trigger warnings or errors.
    """
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        # Use sklearn's StratifiedKFold as the core engine
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def split(self, X: X, y: y, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets.
        
        Args:
            X: The feature dataset (WhaleDataset).
            y: Target whale IDs.
            groups: Not used, maintained for interface compatibility.
            
        Returns:
            An iterator of (train_indices, validation_indices) tuples.
        """
        # Ensure y is a pandas Series for frequency analysis
        if not isinstance(y, pd.Series):
            y_series = pd.Series(y)
        else:
            y_series = y
            
        # Step 1: Analyze class frequencies
        counts = y_series.value_counts()
        
        # Step 2: Identify rare classes based on the technical specification (threshold < 5)
        # Note: self.n_splits is 5 as per the objective requirements.
        rare_classes = counts[counts < 5].index
        
        # Step 3: Create a stratification target by binning rare classes
        # This prevents StratifiedKFold from encountering classes with fewer members than splits
        y_stratify = y_series.copy()
        y_stratify[y_series.isin(rare_classes)] = "rare_class_bin"
        
        # Step 4: Delegate to StratifiedKFold using the binned target
        # We pass a dummy array of indices for X as StratifiedKFold only relies on len(X) and y
        return self.skf.split(np.arange(len(y_series)), y_stratify)

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a Binned Stratified K-Fold splitting strategy.
    
    This strategy is Specifically designed for the Happy Whale identification task's 
    long-tail distribution, ensuring that stratification is performed on a modified 
    target where rare samples are grouped together.
    
    Args:
        X (X): The full training features (WhaleDataset).
        y (y): The full training targets (whale IDs).

    Returns:
        BinnedStratifiedKFold: A splitter object that implements split() and get_n_splits().
    """
    # Technical Specification: n_splits=5, Rare Class Handling < 5
    return BinnedStratifiedKFold(n_splits=5, random_state=42)
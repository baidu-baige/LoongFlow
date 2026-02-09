import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Iterator, Tuple

# Task-adaptive type definitions for PetFinder Pawpularity task
X = pd.DataFrame  # Features: image paths and metadata
y = pd.DataFrame  # Targets: Pawpularity_scaled and metadata labels

class PawpularityStratifiedSplitter:
    """
    Implements a Stratified K-Fold cross-validation strategy for continuous targets.
    
    The strategy bins the continuous Pawpularity score into discrete classes 
    using Sturges' rule to enable stratification, ensuring that each fold 
    maintains a representative distribution of the target variable.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.skf = StratifiedKFold(
            n_splits=self.n_splits, 
            shuffle=self.shuffle, 
            random_state=self.random_state
        )

    def split(self, X: X, y: y, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets.
        
        Args:
            X (pd.DataFrame): Training features.
            y (pd.DataFrame): Training targets containing 'Pawpularity_scaled'.
            
        Returns:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Training and validation indices.
        """
        # Extract the primary target for stratification
        # Pawpularity_scaled is the normalized target [0.01, 1.0]
        target = y['Pawpularity_scaled']
        n_samples = len(target)
        
        # Calculate number of bins using Sturges' rule: k = ceil(log2(n) + 1)
        # This creates approximately 10-15 bins for the dataset sizes involved (200 to 8920)
        num_bins = int(np.ceil(np.log2(n_samples) + 1))
        
        # Discretize the target into bins to allow StratifiedKFold to operate
        # labels=False returns integer codes for each bin
        bins = pd.cut(target, bins=num_bins, labels=False)
        
        # Perform stratified split based on the generated bins
        return self.skf.split(X, bins)

    def get_n_splits(self, X: X = None, y: y = None, groups=None) -> int:
        """
        Returns the number of folds.
        """
        return self.n_splits

def get_splitter(X: X, y: y) -> PawpularityStratifiedSplitter:
    """
    Defines and returns the data partitioning strategy for validation.

    Args:
        X (pd.DataFrame): The full training features.
        y (pd.DataFrame): The full training targets.

    Returns:
        PawpularityStratifiedSplitter: Configured splitter instance using binned stratification.
    """
    # 5-fold stratification ensures 80/20 train/val split per fold
    # Shuffle is enabled with a fixed random state for reproducibility
    return PawpularityStratifiedSplitter(n_splits=5, shuffle=True, random_state=42)
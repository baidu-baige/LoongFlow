from typing import Any, Iterator, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Path constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/3c8d5cca-ccb7-4c25-92f1-7d0f571dedc1/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing tabular and structural data
y = pd.DataFrame      # Target vector for formation energy and bandgap

class NOMADStratifiedSplitter:
    """
    Custom splitter implementing Stratified K-Fold for the NOMAD2018 task.
    Stratification is performed on a synthetic column combining binned bandgap energy 
    and the material's spacegroup to ensure representative splits.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: pd.DataFrame, y: pd.DataFrame, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation set.
        """
        # Validation of required columns for stratification
        if 'bandgap_energy_ev' not in y.columns:
            raise KeyError("Target 'bandgap_energy_ev' required for stratification not found in y.")
        if 'spacegroup' not in X.columns:
            raise KeyError("Feature 'spacegroup' required for stratification not found in X.")

        # Create synthetic stratification labels per technical specification
        # 1. Bin bandgap_energy_ev into deciles (10 bins)
        # Using qcut ensures bins are of approximately equal size
        try:
            bandgap_bins = pd.qcut(y['bandgap_energy_ev'], q=10, labels=False, duplicates='drop')
        except ValueError as e:
            # Fallback if qcut fails due to data distribution, though highly unlikely given EDA
            raise RuntimeError(f"Failed to create quintile bins for stratification: {e}")
            
        # 2. Concatenate with spacegroup to create unique strata
        # This ensures each fold has a representative distribution of both symmetry and target range
        stratify_labels = X['spacegroup'].astype(str) + "_" + bandgap_bins.astype(str)
        
        # Delegate splitting logic to sklearn's StratifiedKFold
        return self.skf.split(X, stratify_labels)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns the data splitting strategy.
    
    Args:
        X (pd.DataFrame): The training features.
        y (pd.DataFrame): The training targets containing formation_energy_ev_natom and bandgap_energy_ev.

    Returns:
        NOMADStratifiedSplitter: A splitter object following the technical specification.
    """
    # Initialize the stratified k-fold strategy as specified
    # n_splits=5, shuffle=True, random_state=42
    splitter = NOMADStratifiedSplitter(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    return splitter
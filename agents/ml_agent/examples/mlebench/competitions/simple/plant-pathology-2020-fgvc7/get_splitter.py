from typing import Any, List, Iterator, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/fc83b0b0-0bd1-41b2-8cd2-ac42c45cd457/2/executor/output"

# Task-adaptive type definitions per load_data implementation
X = List[str]      # Paths to training images
y = np.ndarray    # Multi-label target matrix of shape (N, 4)

class PlantPathologySplitter:
    """
    A validation splitter implementing StratifiedKFold for the Plant Pathology 2020 task.
    Converts multi-column target matrices into a categorical vector to ensure 
    stratification across all classes, particularly for the rare 'multiple_diseases' class.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: X, y: y, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and validation sets.
        
        Args:
            X (X): Feature matrix (List of image paths).
            y (y): Target matrix (N, 4).
            groups (Any): Group labels for the samples used while splitting the dataset.

        Returns:
            Iterator: Yields (train_idx, val_idx) for each fold.
        """
        # Convert multi-label matrix to 1D categorical array for stratification.
        # EDA confirms that the target labels (healthy, multiple_diseases, rust, scab) 
        # are mutually exclusive with no multi-label rows.
        y_categorical = np.argmax(y, axis=1)
        
        # Delegate splitting to StratifiedKFold using the categorical representation
        return self.skf.split(X, y_categorical, groups)

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """
        Returns the number of splitting iterations.
        """
        return self.n_splits

def get_splitter(X: X, y: y) -> PlantPathologySplitter:
    """
    Defines and returns the data splitting strategy for the Plant Pathology task.
    
    This strategy uses StratifiedKFold with 5 splits to ensure that the 
    'multiple_diseases' class (5.2% frequency) is consistently represented 
    in both training and validation sets throughout cross-validation.

    Args:
        X (X): The full training features (image paths).
        y (y): The full training targets (one-hot/multi-label matrix).

    Returns:
        PlantPathologySplitter: A configured splitter object with split() and get_n_splits() methods.
    """
    print("Stage 2: Defining StratifiedKFold splitter...")
    print(f"Parameters: n_splits=5, shuffle=True, random_state=42")
    
    # Initialize splitter according to Technical Specification
    splitter = PlantPathologySplitter(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    return splitter
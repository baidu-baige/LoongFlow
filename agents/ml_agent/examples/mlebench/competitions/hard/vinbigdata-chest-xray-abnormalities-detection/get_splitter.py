import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Iterator, Tuple, Any

# Task-adaptive type definitions 
# X: Feature matrix containing metadata and filepaths for each unique image.
# y: List of 2D numpy arrays, each containing bounding boxes and class IDs for an image.
X = pd.DataFrame
y = List[np.ndarray]

class VinBigDataSplitter:
    """
    Validation strategy: Stratified GroupKFold.
    
    This splitter combines multi-label stratification with image-level grouping.
    - Grouping: Prevents leakage by ensuring all boxes for a single image_id stay within the same fold.
    - Stratification: Ensures balanced distribution of rare thoracic abnormality classes (0-13)
      across folds by creating a composite label from the multi-label indicator matrix.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        # StratifiedGroupKFold requires scikit-learn >= 1.0
        self.sgkf = StratifiedGroupKFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )

    def split(self, X: pd.DataFrame, y: List[np.ndarray], groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates train/validation indices.
        
        Args:
            X: Dataframe where each row is a unique image.
            y: List of numpy arrays [x_min, y_min, x_max, y_max, class_id].
            groups: Optional grouping variable (defaults to X['image_id']).
            
        Returns:
            Iterator of (train_idx, val_idx).
        """
        if y is None:
            raise ValueError("Target 'y' must be provided for stratified splitting.")
        
        # 1. Construct multi-label indicator matrix (14 abnormality classes: 0-13)
        # We ignore class 14 (No finding) in the indicator as it is the complement of abnormalities.
        num_classes = 14
        n_samples = len(y)
        indicators = np.zeros((n_samples, num_classes), dtype=np.int8)
        
        for i, boxes in enumerate(y):
            if boxes.size > 0:
                # Extract class_ids from the 5th column of the box array
                classes = boxes[:, 4].astype(int)
                for c in classes:
                    if 0 <= c < num_classes:
                        indicators[i, c] = 1
        
        # 2. Convert multi-label indicators to a 1D target for StratifiedGroupKFold
        # Each unique combination of findings is mapped to a distinct string label.
        y_stratify = np.array(["".join(map(str, row)) for row in indicators])
        
        # 3. Define groups based on image_id to prevent data leakage
        if groups is None:
            if 'image_id' not in X.columns:
                raise KeyError("Column 'image_id' not found in X; required for grouping.")
            groups = X['image_id'].values
            
        return self.sgkf.split(X, y_stratify, groups=groups)

    def get_n_splits(self) -> int:
        """Returns the number of folds."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns the Stratified GroupKFold data splitting strategy.

    Args:
        X (pd.DataFrame): The full training features/metadata.
        y (List[np.ndarray]): The full training targets (bounding boxes and classes).

    Returns:
        VinBigDataSplitter: A splitter object providing 5-fold cross-validation.
    """
    # Parameters initialized as per technical specification
    return VinBigDataSplitter(n_splits=5, shuffle=True, random_state=42)
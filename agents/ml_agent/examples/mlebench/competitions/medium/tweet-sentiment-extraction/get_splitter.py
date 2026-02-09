import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Any, Iterator, Tuple

# Task-adaptive type definitions
X = pd.DataFrame    # Feature matrix containing 'text' and 'sentiment'
y = pd.Series       # Target vector containing 'selected_text'

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/tweet-sentiment-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/1cc1b6ac-193c-4f3f-a388-380b832f53e8/5/executor/output"

class SentimentStratifiedSplitter:
    """
    A custom splitter that enforces stratification on the 'sentiment' column 
    of the input features.
    
    This ensures consistent sentiment distribution across training and validation 
    folds, which is critical for stable Jaccard score estimation given the 
    skewed sentiment distribution (Neutral: ~40%, Positive: ~31%, Negative: ~28%).
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        # StratifiedKFold is efficient for the dataset size (~25k samples).
        # We use the CPU implementation as index generation is not a bottleneck.
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: pd.DataFrame, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and validation sets.
        
        Args:
            X (pd.DataFrame): Training features containing the 'sentiment' column.
            y (Any): Ignored (stratification uses 'sentiment' from X).
            groups (Any): Ignored.
            
        Returns:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Train and validation indices.
        """
        if 'sentiment' not in X.columns:
            raise KeyError("The feature matrix X must contain a 'sentiment' column for stratification.")
        
        # Explicitly stratify on X['sentiment'] to preserve class distribution across folds.
        # We pass X as the arrays and X['sentiment'] as the groups to stratify by.
        return self.skf.split(X, X['sentiment'])

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    This implementation reuses the parent logic to perform a 5-fold Stratified K-Fold 
    cross-validation based on the 'sentiment' category.

    Args:
        X (X): The full training features (pd.DataFrame).
        y (y): The full training targets (pd.Series).

    Returns:
        Any: A SentimentStratifiedSplitter instance.
    """
    # Configuration:
    # - Method: StratifiedKFold (via custom wrapper)
    # - n_splits: 5
    # - shuffle: True
    # - random_state: 42
    # - Stratification Target: 'sentiment' column in X
    
    splitter = SentimentStratifiedSplitter(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    return splitter
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Any, Iterator, Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/chaii-hindi-and-tamil-question-answering/prepared/public"
OUTPUT_DATA_PATH = "output/af0f7d71-a062-46e3-8926-51aedd28d3b4/3/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix type: contains 'id', 'context', 'question', 'language'
y = pd.DataFrame  # Target vector type: contains 'answer_text', 'answer_start'

class LanguageStratifiedSplitter:
    """
    A validation strategy wrapper that ensures stratification based on the 'language' column.
    Maintains consistent language distribution (approx. 66% Hindi / 34% Tamil) across 5 folds.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.cv = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[Any, Any]]:
        """
        Generates indices to split data into training and validation sets.
        
        Args:
            X (pd.DataFrame): Training features containing the 'language' column.
            y (pd.DataFrame, optional): Training targets.
            groups (Any, optional): Group labels for the samples.

        Returns:
            Iterator[Tuple[np.ndarray, np.ndarray]]: The training and validation indices.
        """
        if 'language' not in X.columns:
            raise KeyError("The 'language' column is missing in X. It is required for stratified splitting.")
        
        # Stratify specifically on the language labels
        return self.cv.split(X, X['language'])

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

def get_splitter(X: X, y: y) -> LanguageStratifiedSplitter:
    """
    Defines and returns a data splitting strategy for model validation.

    This function configures a Stratified 5-Fold cross-validation strategy 
    to ensure reliable evaluation across Hindi and Tamil subsets.

    Args:
        X (X): The full training features.
        y (y): The full training targets.

    Returns:
        LanguageStratifiedSplitter: A splitter object that implements split() and get_n_splits().
    """
    print("Initializing Stratified 5-Fold splitter (stratification target: 'language')")
    
    return LanguageStratifiedSplitter(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
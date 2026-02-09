from typing import Any, Iterator, Tuple
import pandas as pd
from sklearn.model_selection import GroupKFold

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing 'file_path' and 'speaker_id'
y = pd.Series         # Target vector (labels)

class SpeakerGroupSplitter:
    """
    A validation strategy implementation using GroupKFold to prevent speaker leakage.
    Ensures that multiple utterances from the same speaker do not appear in both 
    training and validation sets, mimicking the test environment where speakers are unseen.
    """
    def __init__(self, n_splits: int = 5):
        """
        Initializes the splitter with the specified number of folds.
        
        Args:
            n_splits (int): Number of folds for cross-validation. Default is 5.
        """
        self.n_splits = n_splits
        self.cv = GroupKFold(n_splits=n_splits)

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[Any, Any]]:
        """
        Generates indices to split data into training and validation sets using 'speaker_id'.
        
        Args:
            X (pd.DataFrame): Training features containing 'speaker_id'.
            y (pd.Series): Training labels.
            groups (Any): Optional grouping array. If None, 'speaker_id' column is used.
            
        Returns:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Train and validation indices.
        """
        # technical specification: Target: Grouping by speaker_id
        if groups is None:
            if isinstance(X, pd.DataFrame) and 'speaker_id' in X.columns:
                groups = X['speaker_id']
            else:
                raise KeyError("The input feature matrix X must contain a 'speaker_id' column for GroupKFold splitting.")
        
        # sklearn GroupKFold does not use random_state; it is deterministic based on group order
        return self.cv.split(X, y, groups=groups)

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """
        Returns the number of splitting iterations.
        """
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns the data partitioning strategy for the speech recognition task.

    This function sets up a GroupKFold strategy to ensure that validation performance
    is a reliable proxy for test performance by preventing speaker-level information leakage.

    Args:
        X (X): The full training features (DataFrame containing 'speaker_id').
        y (y): The full training targets.

    Returns:
        SpeakerGroupSplitter: A splitter object implementing split() and get_n_splits().
    """
    # Technical Specification: 
    # Method: GroupKFold
    # Target: grouping by speaker_id
    # Parameters: n_splits=5
    return SpeakerGroupSplitter(n_splits=5)
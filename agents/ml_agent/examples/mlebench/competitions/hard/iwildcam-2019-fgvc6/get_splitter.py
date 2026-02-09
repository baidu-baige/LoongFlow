from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Define concrete types for this task
X = pd.DataFrame
y = pd.Series

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.
    
    This implementation uses a grouping strategy based on camera locations to prevent 
    data leakage, ensuring that the model generalizes to new geographical regions 
    as required by the iWildCam 2019 task. 

    Args:
        X (pd.DataFrame): The full training features containing the 'location' column.
        y (pd.Series): The full training targets for stratification.

    Returns:
        Any: A splitter object implementing split() and get_n_splits().
    """
    
    # Technical Specification:
    # - Method: GroupKFold (Implemented via StratifiedGroupKFold to support shuffle and random_state)
    # - Target: 'location' column
    # - Parameters: n_splits=5, shuffle=True, random_state=42
    
    class LocationGroupSplitter:
        """
        A wrapper around StratifiedGroupKFold that automatically uses the 'location' 
        column as the grouping key if no groups are explicitly provided.
        """
        def __init__(self, n_splits: int, shuffle: bool, random_state: int):
            self.n_splits = n_splits
            # StratifiedGroupKFold is chosen because it respects the 'groups' constraint 
            # while allowing for 'shuffle' and 'random_state', and it preserves 
            # class distributions across folds which is critical for imbalanced data.
            self.cv = StratifiedGroupKFold(
                n_splits=n_splits, 
                shuffle=shuffle, 
                random_state=random_state
            )
            
        def split(self, X: pd.DataFrame, y: pd.Series, groups=None):
            """
            Generates indices to split data into training and validation sets.
            """
            # Ensure information leakage is prevented by using 'location' as groups.
            # If the downstream workflow does not pass groups, we extract 'location' from X.
            if groups is None:
                if isinstance(X, pd.DataFrame) and 'location' in X.columns:
                    groups = X['location']
                else:
                    # Propagate error if the required grouping column is missing
                    raise ValueError(
                        "Grouping variable required for iWildCam validation strategy. "
                        "Ensure 'location' column is present in X or pass groups directly."
                    )
            
            return self.cv.split(X, y, groups=groups)
            
        def get_n_splits(self, X=None, y=None, groups=None) -> int:
            """Returns the number of splitting iterations."""
            return self.n_splits

    # Initialize and return the splitter
    return LocationGroupSplitter(n_splits=5, shuffle=True, random_state=42)
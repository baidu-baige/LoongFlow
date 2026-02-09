import pandas as pd
from typing import Union
from sklearn.model_selection import StratifiedKFold

# Concrete type definitions for this task
X = pd.DataFrame  # Features: contains id, prompt, response_a, response_b
y = pd.Series     # Targets: single integer column [0, 1, 2]

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    This function implements a StratifiedKFold strategy to ensure each fold is 
    representative of the global class distribution (winner_model_a, winner_model_b, winner_tie),
    which is critical for the LMSYS Chatbot Arena multi-class classification task.

    Args:
        X (X): The full training features (pd.DataFrame).
        y (y): The full training targets (pd.Series) mapping to [0, 1, 2].

    Returns:
        StratifiedKFold: A splitter object that implements:
            - split(X, y=None, groups=None) -> Iterator[(train_idx, val_idx)]
            - get_n_splits() -> int
    """
    # Step 1: Configure StratifiedKFold based on technical specifications
    # n_splits=5: Standard 5-fold cross-validation
    # shuffle=True: Ensure data is shuffled before splitting
    # random_state=42: Ensure reproducibility of folds
    splitter = StratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )

    # Step 2: Return the configured splitter instance
    # The splitter will be used downstream by the workflow to generate (train_idx, val_idx)
    return splitter
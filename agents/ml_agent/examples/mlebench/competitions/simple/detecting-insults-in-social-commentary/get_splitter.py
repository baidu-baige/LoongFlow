from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/f0abd8d6-b251-4e86-b7b3-c7603506ee1b/1/executor/output"

# Task-adaptive type definitions
# X: Feature matrix containing 'Comment' and 'Date' attributes
# y: Target vector containing binary 'Insult' labels
X = pd.DataFrame
y = pd.Series

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    This function determines HOW to partition data for training vs validation.
    Given the small dataset size (3947 samples) and class imbalance (26.6% positive),
    StratifiedKFold is utilized to ensure that each fold is representative of the
    overall class distribution, reducing metric variance and preventing leakage.

    Args:
        X (X): The full training features (DataFrame).
        y (y): The full training targets (Series).

    Returns:
        StratifiedKFold: A scikit-learn splitter object configured with:
            - n_splits: 5
            - shuffle: True
            - random_state: 42
    """
    # Step 1: Analyze task type and data characteristics
    # Task: Binary classification with class imbalance.
    # Data: Small sample size (N < 5000) prone to variance.

    # Step 2: Select appropriate splitter based on analysis
    # StratifiedKFold maintains the percentage of samples for each class.
    # Shuffle is enabled to handle any potential ordering in the raw data.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 3: Return configured splitter instance
    return splitter
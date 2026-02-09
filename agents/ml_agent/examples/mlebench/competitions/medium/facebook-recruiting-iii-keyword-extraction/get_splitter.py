from typing import Iterator, Tuple
import pandas as pd
from sklearn.model_selection import ShuffleSplit

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix type: Title and Body
y = pd.Series         # Target vector type: Tags

def get_splitter(X: X, y: y) -> ShuffleSplit:
    """
    Defines and returns a data splitting strategy for model validation.

    This function configures a ShuffleSplit strategy for the Facebook Recruiting III task.
    With approximately 5.4 million rows (after deduplication in load_data), a single 
    90/10 split is statistically sufficient to provide a stable validation metric 
    while maximizing training data.

    Args:
        X (X): The full training features (Title, Body).
        y (y): The full training targets (Tags).

    Returns:
        ShuffleSplit: A splitter object configured for a single 90/10 split.
    """
    # Step 1: Analyze task type and data characteristics
    # The dataset size is large (5.4M rows). Cross-validation (e.g., KFold) would be 
    # computationally expensive and likely unnecessary for variance reduction. 
    # A single hold-out set is appropriate.

    # Step 2: Select and configure the splitter
    # We use ShuffleSplit to ensure the 10% validation set is randomly sampled 
    # from the deduplicated training set.
    splitter = ShuffleSplit(
        n_splits=1, 
        test_size=0.1, 
        random_state=42
    )

    # Step 3: Return configured splitter instance
    return splitter
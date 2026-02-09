import pandas as pd
from typing import Any
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/4d08636e-bf37-40e0-b9d7-8ffb77d57ea2/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix: contains 'comment_text'
y = pd.DataFrame  # Target vector: contains 6 toxicity labels

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a MultilabelStratifiedKFold validation strategy.

    This strategy ensures that the distribution of multiple labels is preserved across folds, 
    which is essential for this task given the significant class imbalance and sparsity 
    in labels like 'threat' (0.3%) and 'identity_hate' (0.88%).

    Args:
        X (X): The full training features.
        y (y): The full training targets (multi-label binary matrix).

    Returns:
        MultilabelStratifiedKFold: A splitter object that implements .split(X, y) 
                                   and .get_n_splits().
    """
    # MultilabelStratifiedKFold from iterative-stratification is used to maintain 
    # the proportions of all 6 target classes in each fold.
    splitter = MultilabelStratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    return splitter
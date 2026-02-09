from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/f2dbb22d-a0cb-4add-aa87-f2c6b1a4b76f/77/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame    # Feature matrix type 
y = pd.Series       # Target vector type

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    This function implements a 5-fold Stratified K-Fold cross-validation strategy.
    Stratification is critical for this task due to the class imbalance (approx. 25% 
    success rate), ensuring that each fold represents the overall target distribution 
    to provide a stable estimate of the AUC-ROC metric.

    Args:
        X (X): The full training features.
        y (y): The full training targets.

    Returns:
        StratifiedKFold: A configured splitter object for cross-validation with 
                         .split() and .get_n_splits() methods.
    """
    # Step 1: Analyze task type and data characteristics
    # The dataset contains 2878 samples with a binary target. 
    # AUC-ROC is sensitive to class distribution in validation sets. 
    # Stratification ensures consistency across folds and prevents metric variance.

    # Step 2: Select appropriate splitter based on analysis
    # Technical Specification: Reuse parent's logic.
    # n_splits=5: Provides ~575 samples per validation fold, sufficient for stable AUC.
    # shuffle=True: Removes any potential ordering bias present in the source JSON.
    # random_state=42: Ensures deterministic splits for reproducibility across pipeline runs.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 3: Return configured splitter instance
    return splitter
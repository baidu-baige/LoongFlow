import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Union

# Task-adaptive type definitions
# X: Feature matrix containing the raw text snippets.
# y: Target vector containing the categorical author labels.
X = pd.DataFrame
y = pd.Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/spooky-author-identification/prepared/public"
OUTPUT_DATA_PATH = "output/368bc6e8-482c-48b8-a870-040b0c3a264c/6/executor/output"

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    For the Spooky Author Identification task, we utilize a 5-fold StratifiedKFold 
    strategy. Given the sample size of ~17.6k and the multi-class classification 
    nature (EAP, HPL, MWS), stratification is essential to maintain consistent class 
    frequencies across folds, ensuring that the log-loss evaluation is reliable.

    Args:
        X (X): The full training features (DataFrame).
        y (y): The full training targets (Series).

    Returns:
        StratifiedKFold: A configured splitter object that implements split() and get_n_splits().
    """
    # Step 1: Analyze task type and data characteristics
    # Task: Multi-class classification (3 authors).
    # Data Size: ~17.6k samples.
    # Distribution: EAP (40.2%), MWS (31.0%), HPL (28.8%).

    # Step 2: Select appropriate splitter
    # StratifiedKFold prevents class distribution shifts, which is critical for
    # stable multi-class log-loss optimization and evaluation.
    
    # Validation of inputs to ensure alignment before initializing the splitter.
    if X is None or y is None:
        raise ValueError("Input features X and targets y must not be None.")
    
    if len(X) != len(y):
        raise ValueError(f"Alignment Error: Feature count ({len(X)}) does not match target count ({len(y)}).")

    # Step 3: Configure and return the splitter instance
    # - n_splits=5: Standard choice for ~20k samples to balance validation stability and training time.
    # - shuffle=True: Essential to remove any potential sequence-based bias in the source data.
    # - random_state=42: Guaranteed reproducibility for the entire ML pipeline.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    return splitter
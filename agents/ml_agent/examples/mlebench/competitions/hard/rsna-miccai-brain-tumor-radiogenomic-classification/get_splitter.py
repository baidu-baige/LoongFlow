from typing import Any
import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/rsna-miccai-brain-tumor-radiogenomic-classification/prepared/public"
OUTPUT_DATA_PATH = "output/dafb557f-655e-4395-9835-6f75549a5b27/1/executor/output"

# Task-adaptive type definitions
# X: 5D NumPy array (N_samples, C_modalities, Depth, Height, Width)
# y: 1D NumPy array (N_samples,) representing the MGMT_value target
X = np.ndarray
y = np.ndarray

def get_splitter(X: X, y: y) -> StratifiedKFold:
    """
    Defines and returns a data splitting strategy for model validation.

    This function implements a 5-fold Stratified Cross-Validation strategy.
    Given the small sample size (n=526) and nearly balanced classes, stratification 
    is essential to ensure that each fold maintains a consistent distribution of the 
    MGMT_value target. Since the input X is already aggregated at the patient level 
    (one 5D volume per BraTS21ID), StratifiedKFold inherently prevents patient-level 
    data leakage while providing robust evaluation.

    Args:
        X (X): The full training features.
        y (y): The full training targets.

    Returns:
        StratifiedKFold: A splitter object that implements:
            - split(X, y=None, groups=None) -> Iterator[(train_idx, val_idx)]
            - get_n_splits() -> int
    """
    # Implementation follows technical specification:
    # Method: StratifiedKFold
    # Parameters: n_splits=5, shuffle=True, random_state=42
    splitter = StratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    return splitter
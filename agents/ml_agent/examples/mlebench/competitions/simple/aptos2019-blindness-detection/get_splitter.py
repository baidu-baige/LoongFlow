from typing import Any
from sklearn.model_selection import StratifiedKFold
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/aptos2019-blindness-detection/prepared/public"
OUTPUT_DATA_PATH = "output/16326c74-72ad-4b59-ad28-cc76a3d9d373/5/executor/output"

# Task-adaptive type definitions
X = pd.Series   # Series containing full string paths to image files
y = pd.Series  # Series containing diagnosis labels as float32 (ordinal 0.0-4.0)

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    The APTOS 2019 dataset exhibits significant class imbalance (approx. 50% Class 0).
    A Stratified K-Fold approach ensures that each fold maintains the same proportion 
    of severity levels (0-4), which is critical for the reliable calculation of 
    the Quadratic Weighted Kappa (QWK) metric and for ensuring the model encounters 
    rare classes (3 and 4) in every training split.

    Args:
        X (X): The full training data (image paths).
        y (y): The full training targets (diagnosis labels).

    Returns:
        StratifiedKFold: A splitter object configured for 5-fold stratified cross-validation.
            Implements .split(X, y) and .get_n_splits().
    """
    
    # Step 1: Analyze task type and data characteristics
    # The dataset has 3,295 samples with 5 discrete ordinal classes.
    # Class distribution is skewed, making stratification necessary.
    
    # Step 2: Select appropriate splitter based on analysis
    # StratifiedKFold is standard for classification/ordinal-regression tasks with imbalance.
    # n_splits=5 provides an 80/20 train/validation ratio.
    # random_state=42 ensures the splits are reproducible across experiments.
    splitter = StratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=42
    )
    
    # Step 3: Return configured splitter instance
    # Downstream workflow will call splitter.split(data, targets).
    # Even though targets are float32, StratifiedKFold handles them as discrete 
    # labels if they represent distinct whole numbers (0.0, 1.0, ..., 4.0).
    return splitter
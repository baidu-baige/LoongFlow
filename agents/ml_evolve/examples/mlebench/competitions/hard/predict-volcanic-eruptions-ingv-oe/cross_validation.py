# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/43637237-1f49-4750-a868-8602ac177881/1/executor/output"

# Type Definitions
# Features: pd.DataFrame containing segment_id and file paths
# Labels: pd.Series containing time_to_eruption
Features = pd.DataFrame
Labels = pd.Series


def cross_validation(X: Features, y: Labels) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.
    
    Args:
        X (Features): The full training data features (metadata/paths).
        y (Labels): The full training data labels (time_to_eruption).
    
    Returns:
        BaseCrossValidator: An instance of a 5-Fold K-Fold cross-validation splitter.
    """
    # Step 1: Analyze task type and data characteristics
    # The dataset consists of independent 10-minute seismic segments. 
    # The target 'time_to_eruption' is a continuous numeric value.
    # Since there is no explicit temporal grouping or hierarchy (like multiple segments per eruption event) 
    # provided in the metadata, standard K-Fold is the most robust starting strategy.

    # Step 2: Select appropriate splitter based on analysis
    # 5-Fold K-Fold provides a standard 80/20 train/validation split.
    # Shuffling is enabled to prevent any ordering bias and ensure a representative 
    # target distribution across folds.
    splitter = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 3: Return configured splitter instance
    return splitter

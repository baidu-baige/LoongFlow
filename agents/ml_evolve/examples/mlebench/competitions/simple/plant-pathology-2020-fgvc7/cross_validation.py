# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/1799683e-18f2-43a3-97d1-8b0bdddc3200/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class OneHotStratifiedKFold(StratifiedKFold):
    """
    Custom StratifiedKFold implementation that handles one-hot encoded target variables
    by collapsing them into a single integer label for stratification purposes.
    """

    def split(self, X, y, groups=None):
        # Implementation Guidance: Create a single integer label for stratification 
        # by finding the index of the '1'. This handles the mutually exclusive nature 
        # of the Plant Pathology dataset targets.
        y_strat = y.values.argmax(axis=1) if hasattr(y, 'values') else y
        return super().split(X, y_strat, groups)


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a StratifiedKFold cross-validation splitter.
    
    Args:
        X (DT): The full training data features (image paths). 
        y (DT): The full training data labels (one-hot encoded).
    
    Returns:
        BaseCrossValidator: An instance of the custom OneHotStratifiedKFold splitter.
    """
    # Step 1: Analyze task type and data characteristics
    # The dataset is small (1638 samples) and imbalanced, particularly the 'multiple_diseases' class (85 samples).
    # Stratified CV is essential to ensure stable ROC AUC estimates.

    # Step 2: Select appropriate splitter based on analysis
    # StratifiedKFold with n_splits=5, shuffle=True, and random_state=42 provides a robust balance 
    # between validation reliability and computational efficiency.

    # Step 3: Return configured splitter instance
    # We use a custom subclass to ensure the splitter correctly processes the one-hot target DataFrame.
    return OneHotStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

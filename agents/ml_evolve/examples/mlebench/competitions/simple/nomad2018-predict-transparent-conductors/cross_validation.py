# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/2ea9a3e6-0185-40d8-bd93-7afe244a50a1/2/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def cross_validation(X: DT, y: DT) -> BaseCrossValidator:
    """
    Defines and returns a scikit-learn style cross-validation splitter.

    Args:
        X (DT): The full training data features. Useful for checking if 'Group' or 'Time' columns exist.
        y (DT): The full training data labels. Useful for checking class distribution (Stratified).

    Returns:
        BaseCrossValidator: An instance of a cross-validation splitter 
                            (e.g., KFold, StratifiedKFold, TimeSeriesSplit).
    """
    # Step 1: Analyze the task type
    # This is a regression task with two continuous targets:
    # - formation_energy_ev_natom
    # - bandgap_energy_ev
    # The evaluation metric is RMSLE (Root Mean Squared Logarithmic Error)

    # Step 2: Check for special constraints
    # Based on EDA analysis:
    # - No time-series structure in the data
    # - No explicit group structure that needs to be preserved
    # - Data consists of 2160 training samples with 6 unique spacegroups
    # - Targets have negative correlation (-0.45), suggesting independent evaluation

    # Step 3: Instantiate the appropriate splitter
    # For regression tasks without special constraints, standard KFold is appropriate
    # Using shuffle=True to ensure random distribution across folds
    # Using random_state=42 for reproducibility
    # Using n_splits=5 as specified in guidance

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Step 4: Return the splitter object
    return cv

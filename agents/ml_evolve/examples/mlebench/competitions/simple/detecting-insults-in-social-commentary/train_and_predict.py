# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/bb8b0571-64fb-49b6-8e49-1fc2d52da49b/2/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT], Tuple[DT, DT]]


# ===== Training Functions =====

def train_logistic_regression(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains a Logistic Regression model and returns predictions for validation and test sets.

    This function is executed within a cross-validation loop.

    Args:
        X_train (DT): Feature-engineered training set.
        y_train (DT): Training labels.
        X_val (DT): Feature-engineered validation set.
        y_val (DT): Validation labels.
        X_test (DT): Feature-engineered test set.

    Returns:
        Tuple[DT, DT]: A tuple containing:
        - validation_predictions (DT): Predictions for X_val.
        - test_predictions (DT): Predictions for X_test.
    """
    # Step 1: Build and configure model
    # Hyperparameters based on plan: C=1.0, penalty='l2', solver='liblinear', class_weight='balanced'
    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )

    # Step 2: Enable GPU acceleration if supported by the model
    # LogisticRegression in sklearn does not support GPU directly. 
    # Liblinear solver is efficient for this scale on CPU.

    # Step 3: Train on (X_train, y_train)
    model.fit(X_train, y_train)

    # Step 4: Predict on X_val and X_test
    # The task asks for a real number in [0, 1] representing the probability of being an insult.
    validation_predictions = model.predict_proba(X_val)[:, 1]
    test_predictions = model.predict_proba(X_test)[:, 1]

    # Step 5: Return (validation_predictions, test_predictions)
    return validation_predictions, test_predictions


# ===== Model Registry =====
# Register ALL training functions here for the pipeline to use
# Key: Descriptive model name (e.g., "lgbm_tuned", "neural_net")
# Value: The training function reference

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "logistic_regression": train_logistic_regression,
}

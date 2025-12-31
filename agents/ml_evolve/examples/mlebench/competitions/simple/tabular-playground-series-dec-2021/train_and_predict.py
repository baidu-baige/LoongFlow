# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Callable, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/ead961ce-50a1-41ec-89e9-4cc0d527fbe5/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT], Tuple[DT, DT]]


# ===== Training Functions =====

def train_lightgbm(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains a LightGBM model with GPU acceleration and returns class probabilities.

    This function is executed within a cross-validation loop.

    Args:
        X_train (DT): Feature-engineered training set.
        y_train (DT): Training labels.
        X_val (DT): Feature-engineered validation set.
        y_val (DT): Validation labels.
        X_test (DT): Feature-engineered test set.

    Returns:
        Tuple[DT, DT]: A tuple containing:
        - validation_predictions (DT): Probabilities for X_val (n_samples, n_classes).
        - test_predictions (DT): Probabilities for X_test (n_samples, n_classes).
    """
    # Step 1 & 2: Build and configure model with GPU acceleration
    # Note: Using device='cuda' as per hardware guidelines for LightGBM
    params = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'max_depth': -1,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'device': 'cuda',
        'verbosity': -1,
        'n_jobs': -1
    }

    model = lgb.LGBMClassifier(**params)

    # Step 3: Train on (X_train, y_train) with early stopping on (X_val, y_val)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )

    # Step 4: Predict probabilities on X_val and X_test
    # predict_proba returns (n_samples, n_classes)
    validation_predictions = model.predict_proba(X_val)
    test_predictions = model.predict_proba(X_test)

    # Step 5: Return predictions
    # Ensure no NaN or Inf values (not expected from LightGBM with valid input)
    if np.isnan(validation_predictions).any() or np.isinf(validation_predictions).any():
        raise ValueError("Validation predictions contain NaN or Infinity.")
    if np.isnan(test_predictions).any() or np.isinf(test_predictions).any():
        raise ValueError("Test predictions contain NaN or Infinity.")

    return validation_predictions, test_predictions


# ===== Model Registry =====
# Register ALL training functions here for the pipeline to use
# Key: Descriptive model name (e.g., "lgbm_tuned", "neural_net")
# Value: The training function reference

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "lightgbm": train_lightgbm,
}

# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Callable, Dict, Tuple

import lightgbm as lgb
import numpy as np

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/43637237-1f49-4750-a868-8602ac177881/1/executor/output"

# Type Definitions
Features = Any  # pd.DataFrame
Labels = Any  # pd.Series
Predictions = Any  # np.ndarray

PredictionFunction = Callable[
    [Features, Labels, Features, Labels, Features],
    Tuple[Predictions, Predictions]
]


# ===== Training Functions =====

def train_lightgbm(
    X_train: Features,
    y_train: Labels,
    X_val: Features,
    y_val: Labels,
    X_test: Features
) -> Tuple[Predictions, Predictions]:
    """
    Trains a LightGBM model with MAE optimization and GPU acceleration.

    Args:
        X_train (Features): Feature-engineered training set.
        y_train (Labels): Training labels.
        X_val (Features): Feature-engineered validation set.
        y_val (Labels): Validation labels.
        X_test (Features): Feature-engineered test set.

    Returns:
        Tuple[Predictions, Predictions]: (validation_predictions, test_predictions)
    """

    # Step 1 & 2: Build and configure model with GPU acceleration
    # Using device='cuda' as per Guideline 0
    model = lgb.LGBMRegressor(
        n_estimators=10000,
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        max_depth=-1,
        random_state=42,
        objective='mae',
        metric='mae',
        device='cuda',
        n_jobs=-1,
        verbose=-1
    )

    # Step 3: Train on (X_train, y_train) with early stopping on (X_val, y_val)
    # Using current LightGBM callback API for early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    # Step 4: Predict on X_val and X_test
    validation_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    # Validation: Ensure output is finite and contains no NaNs/Infs
    if not np.isfinite(validation_predictions).all() or not np.isfinite(test_predictions).all():
        # Handle non-finite values if they emerge, though LGBM is usually robust
        validation_predictions = np.nan_to_num(validation_predictions, nan=0.0, posinf=0.0, neginf=0.0)
        test_predictions = np.nan_to_num(test_predictions, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 5: Return (validation_predictions, test_predictions)
    return validation_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "lightgbm": train_lightgbm,
}

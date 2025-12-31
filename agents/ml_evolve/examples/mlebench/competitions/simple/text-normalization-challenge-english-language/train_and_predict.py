# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Callable, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/text-normalization-challenge-english-language/prepared/public"
OUTPUT_DATA_PATH = "output/12e29d80-a70c-426a-9331-de3aa1a6ce7c/14/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


# ===== Training Functions =====

def train_lgbm_intent_classifier(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains a high-capacity LightGBM classifier to predict semiotic classes 
    of tokens using GPU acceleration.

    Args:
        X_train (DT): Feature-engineered training set.
        y_train (DT): Training labels (contains 'after' and 'class').
        X_val (DT): Feature-engineered validation set.
        y_val (DT): Validation labels (contains 'after' and 'class').
        X_test (DT): Feature-engineered test set.

    Returns:
        Tuple[DT, DT]: A tuple containing:
        - validation_predictions (DT): Predicted classes for X_val.
        - test_predictions (DT): Predicted classes for X_test.
    """
    # Step 1: Data Preparation
    # Drop identifiers and non-feature columns to prevent leakage or noise
    drop_cols = ['sentence_id', 'token_id']
    X_tr = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_va = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    X_te = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # Extract the target 'class' column for training
    # Based on input structure, y_train is a DataFrame containing 'class'
    y_tr = y_train['class'] if isinstance(y_train, pd.DataFrame) and 'class' in y_train.columns else y_train
    y_va = y_val['class'] if isinstance(y_val, pd.DataFrame) and 'class' in y_val.columns else y_val

    # Step 2: Build and configure model
    # device='cuda' is used for GPU acceleration on NVIDIA A10.
    # num_leaves=255 provides high capacity for complex semiotic rule learning.
    # class_weight='balanced' is critical for the long tail of rare classes (e.g., ADDRESS, FRACTION).
    model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=255,
        min_child_samples=50,
        feature_fraction=0.8,
        class_weight='balanced',
        device='cuda',
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )

    # Step 3: Train with early stopping monitored on validation multi_logloss
    # 50 rounds of early stopping ensures we don't overfit while seeking convergence.
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='multi_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # Step 4: Predict on X_val and X_test
    # Predict returns the most likely class label for downstream transformation logic.
    validation_predictions = model.predict(X_va)
    test_predictions = model.predict(X_te)

    # Step 5: Verification and Return
    if validation_predictions is None or test_predictions is None:
        raise ValueError("Model training or inference produced None values.")

    return validation_predictions, test_predictions


# ===== Model Registry =====
# Register the training functions here for the pipeline to use

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "lgbm_intent_classifier": train_lgbm_intent_classifier,
}

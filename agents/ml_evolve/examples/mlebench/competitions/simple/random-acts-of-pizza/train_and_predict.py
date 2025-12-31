# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

BASE_DATA_PATH = "/root/workspace/evolux-ml/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/ed330620-ed29-4387-b009-fed5bf45c1a8/11/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


def train_xgboost(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT,
    **hyper_params: Any
) -> Tuple[DT, DT]:
    """
    Trains an XGBoost classifier with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        **hyper_params: Additional hyperparameters
        
    Returns:
        Tuple containing:
        - Validation predictions (probability of positive class)
        - Test predictions (probability of positive class)
    """
    # Make copies to avoid modifying original data
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # First convert datetime columns to numeric (unix timestamps)
    datetime_cols = [col for col in X_train.columns
                     if pd.api.types.is_datetime64_any_dtype(X_train[col])]
    for col in datetime_cols:
        X_train[col] = X_train[col].astype(np.int64) // 10 ** 9
        X_val[col] = X_val[col].astype(np.int64) // 10 ** 9
        X_test[col] = X_test[col].astype(np.int64) // 10 ** 9

    # Handle list-type columns by converting them to counts
    list_cols = [col for col in X_train.columns
                 if isinstance(X_train[col].iloc[0], list)
                 if col in X_train.columns]
    for col in list_cols:
        X_train[f'{col}_count'] = X_train[col].apply(len)
        X_val[f'{col}_count'] = X_val[col].apply(len)
        X_test[f'{col}_count'] = X_test[col].apply(len)
        X_train.drop(columns=[col], inplace=True)
        X_val.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)

    # Convert all remaining non-numeric data to numeric
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        # Skip text columns as they should be handled in feature engineering
        if col in ['request_text', 'request_title']:
            continue
        # Factorize categorical columns
        codes, _ = pd.factorize(X_train[col], sort=True)
        X_train[col] = codes
        codes, _ = pd.factorize(X_val[col], sort=True)
        X_val[col] = codes
        codes, _ = pd.factorize(X_test[col], sort=True)
        X_test[col] = codes

    # Fill any remaining missing values with 0
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    # Convert to numpy arrays for XGBoost
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    y_train = y_train.values
    y_val = y_val.values

    # Initialize XGBoost model with specified parameters
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,  # Use all available CPU cores
        eval_metric='auc',
        early_stopping_rounds=50,
        tree_method='hist'  # Use histogram-based algorithm for efficiency
    )

    # Train model with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )

    # Generate predictions
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    # Calculate and print validation score
    val_score = roc_auc_score(y_val, val_preds)
    print(f"XGBoost Validation AUC: {val_score:.4f}")

    # Ensure no NaN or infinite values in predictions
    val_preds = np.nan_to_num(val_preds, nan=0.5, posinf=1.0, neginf=0.0)
    test_preds = np.nan_to_num(test_preds, nan=0.5, posinf=1.0, neginf=0.0)

    return val_preds, test_preds


# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "xgboost": train_xgboost,
}

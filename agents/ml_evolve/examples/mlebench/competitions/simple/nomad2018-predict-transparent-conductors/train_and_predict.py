# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Callable, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/2ea9a3e6-0185-40d8-bd93-7afe244a50a1/2/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


def train_lightgbm(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT,
    **hyper_params: Any
) -> Tuple[DT, DT]:
    """
    Trains LightGBM models for multi-output regression (formation_energy and bandgap_energy).
    
    This is a multi-target regression task with two targets:
    - formation_energy_ev_natom
    - bandgap_energy_ev
    
    The targets have been log1p transformed in create_features, so predictions
    will be in log space and need to be transformed back by the ensemble function.
    """
    # Convert to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values
    else:
        X_train_np = np.array(X_train)

    if isinstance(X_val, pd.DataFrame):
        X_val_np = X_val.values
    else:
        X_val_np = np.array(X_val)

    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values
    else:
        X_test_np = np.array(X_test)

    if isinstance(y_train, pd.DataFrame):
        y_train_np = y_train.values
    else:
        y_train_np = np.array(y_train)

    if isinstance(y_val, pd.DataFrame):
        y_val_np = y_val.values
    else:
        y_val_np = np.array(y_val)

    # Ensure 2D arrays for targets
    if y_train_np.ndim == 1:
        y_train_np = y_train_np.reshape(-1, 1)
    if y_val_np.ndim == 1:
        y_val_np = y_val_np.reshape(-1, 1)

    n_targets = y_train_np.shape[1]

    # LightGBM parameters optimized for this task
    # Based on EDA: 2160 training samples, ~80 features after engineering
    # Using CPU-based training as specified
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': -1,
        'verbose': -1,
        'seed': 42
    }

    # Override with any provided hyperparameters
    lgb_params.update(hyper_params)

    val_preds = np.zeros((X_val_np.shape[0], n_targets))
    test_preds = np.zeros((X_test_np.shape[0], n_targets))

    # Train separate model for each target
    for target_idx in range(n_targets):
        y_train_target = y_train_np[:, target_idx]
        y_val_target = y_val_np[:, target_idx]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_np, label=y_train_target)
        val_data = lgb.Dataset(X_val_np, label=y_val_target, reference=train_data)

        # Train model with early stopping
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        # Generate predictions
        val_preds[:, target_idx] = model.predict(X_val_np, num_iteration=model.best_iteration)
        test_preds[:, target_idx] = model.predict(X_test_np, num_iteration=model.best_iteration)

    # Convert to DataFrames with proper column names
    if isinstance(y_train, pd.DataFrame):
        columns = y_train.columns.tolist()
    else:
        columns = ['formation_energy_ev_natom', 'bandgap_energy_ev']

    val_preds_df = pd.DataFrame(val_preds, columns=columns)
    test_preds_df = pd.DataFrame(test_preds, columns=columns)

    # Ensure no NaN or Infinity values
    val_preds_df = val_preds_df.replace([np.inf, -np.inf], np.nan)
    test_preds_df = test_preds_df.replace([np.inf, -np.inf], np.nan)

    # Fill any NaN with column means
    val_preds_df = val_preds_df.fillna(val_preds_df.mean())
    test_preds_df = test_preds_df.fillna(test_preds_df.mean())

    # Final fallback - fill with 0 if still NaN
    val_preds_df = val_preds_df.fillna(0)
    test_preds_df = test_preds_df.fillna(0)

    return val_preds_df, test_preds_df


def train_xgboost(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT,
    **hyper_params: Any
) -> Tuple[DT, DT]:
    """
    Trains XGBoost models for multi-output regression.
    
    XGBoost provides complementary predictions to LightGBM for ensemble diversity.
    """
    import xgboost as xgb

    # Convert to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values
    else:
        X_train_np = np.array(X_train)

    if isinstance(X_val, pd.DataFrame):
        X_val_np = X_val.values
    else:
        X_val_np = np.array(X_val)

    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values
    else:
        X_test_np = np.array(X_test)

    if isinstance(y_train, pd.DataFrame):
        y_train_np = y_train.values
    else:
        y_train_np = np.array(y_train)

    if isinstance(y_val, pd.DataFrame):
        y_val_np = y_val.values
    else:
        y_val_np = np.array(y_val)

    # Ensure 2D arrays for targets
    if y_train_np.ndim == 1:
        y_train_np = y_train_np.reshape(-1, 1)
    if y_val_np.ndim == 1:
        y_val_np = y_val_np.reshape(-1, 1)

    n_targets = y_train_np.shape[1]

    # XGBoost parameters optimized for this task
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'seed': 42,
        'verbosity': 0
    }

    # Override with any provided hyperparameters
    xgb_params.update(hyper_params)

    val_preds = np.zeros((X_val_np.shape[0], n_targets))
    test_preds = np.zeros((X_test_np.shape[0], n_targets))

    # Train separate model for each target
    for target_idx in range(n_targets):
        y_train_target = y_train_np[:, target_idx]
        y_val_target = y_val_np[:, target_idx]

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_np, label=y_train_target)
        dval = xgb.DMatrix(X_val_np, label=y_val_target)
        dtest = xgb.DMatrix(X_test_np)

        # Train model with early stopping
        evals = [(dtrain, 'train'), (dval, 'valid')]
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=False
        )

        # Generate predictions
        val_preds[:, target_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        test_preds[:, target_idx] = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    # Convert to DataFrames with proper column names
    if isinstance(y_train, pd.DataFrame):
        columns = y_train.columns.tolist()
    else:
        columns = ['formation_energy_ev_natom', 'bandgap_energy_ev']

    val_preds_df = pd.DataFrame(val_preds, columns=columns)
    test_preds_df = pd.DataFrame(test_preds, columns=columns)

    # Ensure no NaN or Infinity values
    val_preds_df = val_preds_df.replace([np.inf, -np.inf], np.nan)
    test_preds_df = test_preds_df.replace([np.inf, -np.inf], np.nan)

    # Fill any NaN with column means
    val_preds_df = val_preds_df.fillna(val_preds_df.mean())
    test_preds_df = test_preds_df.fillna(test_preds_df.mean())

    # Final fallback - fill with 0 if still NaN
    val_preds_df = val_preds_df.fillna(0)
    test_preds_df = test_preds_df.fillna(0)

    return val_preds_df, test_preds_df


# ===== Model Registry =====
# Register production-quality models for the ensemble
# Both LightGBM and XGBoost are well-suited for this tabular regression task
# and provide complementary predictions for ensemble diversity

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
}

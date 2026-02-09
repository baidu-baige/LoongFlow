import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from typing import Tuple, Dict, Callable, Any

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.DataFrame
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Training Functions =====

def train_multi_gbdt_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an ensemble of LightGBM, XGBoost, and CatBoost models for each target.
    Targets are transformed using log1p to optimize for the RMSLE metric.
    
    GPU acceleration is used for XGBoost and CatBoost. LightGBM is restricted to CPU
    due to environment build limitations.
    """
    print(f"Starting Training: Ensemble of GBDTs (LGBM[CPU], XGB[GPU], CatBoost[GPU])")
    
    target_cols = y_train.columns.tolist()
    n_targets = len(target_cols)
    
    # Pre-allocate prediction arrays (in log space)
    val_preds_log = np.zeros((len(X_val), n_targets))
    test_preds_log = np.zeros((len(X_test), n_targets))
    
    # Log-transform targets for RMSLE optimization
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    for i, target in enumerate(target_cols):
        print(f"Processing target: {target} ({i+1}/{n_targets})")
        
        y_tr = y_train_log[target].values
        y_va = y_val_log[target].values
        
        # 1. LightGBM implementation (CPU)
        print(f"  Training LightGBM (CPU) for {target}...")
        lgbm_params = {
            'n_estimators': 2000,
            'learning_rate': 0.03,
            'max_depth': 6,
            'num_leaves': 31,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'cpu', 
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        lgb_model = lgb.LGBMRegressor(**lgbm_params)
        lgb_model.fit(
            X_train, y_tr,
            eval_set=[(X_val, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        lgb_val = lgb_model.predict(X_val)
        lgb_test = lgb_model.predict(X_test)
        
        # 2. XGBoost implementation (GPU)
        # early_stopping_rounds is passed to constructor in newer versions
        print(f"  Training XGBoost (GPU) for {target}...")
        xgb_params = {
            'n_estimators': 2000,
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 100
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train, y_tr,
            eval_set=[(X_val, y_va)],
            verbose=False
        )
        xgb_val = xgb_model.predict(X_val)
        xgb_test = xgb_model.predict(X_test)
        
        # 3. CatBoost implementation (GPU)
        print(f"  Training CatBoost (GPU) for {target}...")
        cb_params = {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'RMSE',
            'task_type': 'GPU',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 100,
            'devices': '0'
        }
        cb_model = cb.CatBoostRegressor(**cb_params)
        cb_model.fit(X_train, y_tr, eval_set=(X_val, y_va))
        cb_val = cb_model.predict(X_val)
        cb_test = cb_model.predict(X_test)
        
        # Ensemble: Simple average of the three models in log-space
        val_preds_log[:, i] = (lgb_val + xgb_val + cb_val) / 3.0
        test_preds_log[:, i] = (lgb_test + xgb_test + cb_test) / 3.0
        
    # Standardize predictions: Inverse transform and non-negative clipping
    val_preds_final = np.maximum(0, np.expm1(val_preds_log))
    test_preds_final = np.maximum(0, np.expm1(test_preds_log))
    
    # Final check for numerical stability
    if np.isnan(val_preds_final).any() or np.isnan(test_preds_final).any():
        raise ValueError("Training process generated non-finite predictions (NaN detected).")
        
    print(f"Training successfully completed for all targets.")
    
    return val_preds_final, test_preds_final

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "gbdt_ensemble": train_multi_gbdt_ensemble,
}
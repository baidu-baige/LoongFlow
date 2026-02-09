import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import concurrent.futures
from typing import Tuple, Dict, Callable, Any

# Task-adaptive type definitions
X = pd.DataFrame           # Preprocessed tabular feature matrix
y = pd.Series              # Target series (time_to_eruption)
Predictions = np.ndarray   # Array of predicted values

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/bdc750a4-f0a3-4926-871d-f9675d7cf1ef/1/executor/output"

# ===== Training Functions =====

def train_lgbm_xgb_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an ensemble of high-capacity gradient boosting models (LightGBM and XGBoost) 
    using all available GPUs in parallel.
    
    LightGBM is assigned to GPU 0 and XGBoost to GPU 1. 
    Both models optimize for Mean Absolute Error (MAE) as required by the task.
    """
    print(f"Starting Training: LGBM and XGBoost Ensemble (Parallel on 2 GPUs)...")

    def _train_lgbm() -> Tuple[Predictions, Predictions]:
        """Trains LightGBM using GPU 0."""
        # Parameters mapped from specification and optimized for GPU usage
        lgb_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'device': 'cuda',
            'gpu_device_id': 0,
            'random_state': 42,
            'n_jobs': 18,
            'verbose': -1
        }
        
        # early_stopping_rounds=100 via callback to avoid API deprecation issues
        model = lgb.LGBMRegressor(n_estimators=10000, **lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200)
            ]
        )
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        return val_preds, test_preds

    def _train_xgb() -> Tuple[Predictions, Predictions]:
        """Trains XGBoost using GPU 1."""
        # Parameters mapped from specification. 
        # early_stopping_rounds moved to constructor to avoid TypeError in fit().
        # tree_method='gpu_hist' and gpu_id=1 for device targeting.
        xgb_params = {
            'objective': 'reg:absoluteerror',
            'tree_method': 'gpu_hist',
            'gpu_id': 1,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 100,
            'random_state': 42,
            'n_jobs': 18
        }
        
        model = xgb.XGBRegressor(n_estimators=10000, **xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=200
        )
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        return val_preds, test_preds

    # Execute training of both models concurrently to utilize both H20 GPUs simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_lgbm = executor.submit(_train_lgbm)
        future_xgb = executor.submit(_train_xgb)
        
        # Propagate exceptions immediately if any model fails
        val_preds_lgb, test_preds_lgb = future_lgbm.result()
        val_preds_xgb, test_preds_xgb = future_xgb.result()

    # Create ensemble by averaging predictions
    val_preds = (val_preds_lgb + val_preds_xgb) / 2.0
    test_preds = (test_preds_lgb + test_preds_xgb) / 2.0
    
    # Final validation ensure no NaNs/Infs
    assert not np.isnan(val_preds).any(), "NaN found in ensemble validation predictions"
    assert not np.isnan(test_preds).any(), "NaN found in ensemble test predictions"
    
    print(f"Ensemble training completed successfully.")
    print(f"LGBM Val MAE: {np.mean(np.abs(val_preds_lgb - y_val)):.2f}")
    print(f"XGB Val MAE: {np.mean(np.abs(val_preds_xgb - y_val)):.2f}")
    print(f"Ensemble Val MAE: {np.mean(np.abs(val_preds - y_val)):.2f}")

    return val_preds, test_preds


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "lgbm_xgb_ensemble": train_lgbm_xgb_ensemble,
}
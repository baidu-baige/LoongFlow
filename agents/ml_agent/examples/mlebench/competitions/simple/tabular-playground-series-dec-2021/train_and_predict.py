from typing import Tuple, Any, Dict, Callable
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/477e9955-ebee-46f4-96b0-878df6f022f5/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Training Functions =====

def train_xgb_gpu(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an XGBoost classifier using GPU acceleration to handle the large-scale 
    Forest Cover Type dataset. Returns class probabilities for ensembling.
    
    Args:
        X_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Training targets (Cover_Type).
        X_val (pd.DataFrame): Preprocessed validation features.
        y_val (pd.Series): Validation targets.
        X_test (pd.DataFrame): Preprocessed test features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (val_probs, test_probs)
    """
    print(f"Initializing XGBoost Training. Train size: {X_train.shape}, Val size: {X_val.shape}")

    # Step 1: Label Encoding
    # XGBoost requires 0-indexed labels for multiclass classification.
    # The target classes are [1, 2, 3, 4, 6, 7] (5 was removed in load_data).
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    num_classes = len(le.classes_)
    print(f"Encoders fitted. Target classes mapped: {le.classes_} to {np.unique(y_train_enc)}")

    # Step 2: Build and configure model
    # Using 'device'='cuda' and 'tree_method'='hist' as per hardware guidelines for modern XGBoost.
    # Parameters derived from Technical Specification.
    model = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.05,
        max_depth=10,
        tree_method='hist',
        device='cuda',
        objective='multi:softprob',
        eval_metric='mlogloss',
        early_stopping_rounds=50,
        random_state=42,
        verbosity=1,
        # Multi-class specific
        n_jobs=-1
    )

    # Step 3: Train with Early Stopping
    print("Starting model training with early stopping...")
    model.fit(
        X_train, 
        y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=100
    )

    # Step 4: Predict probabilities
    print("Generating predictions on validation and test sets...")
    # predict_proba returns an array of shape (n_samples, n_classes)
    val_preds = model.predict_proba(X_val)
    test_preds = model.predict_proba(X_test)

    # Step 5: Post-processing and Validation
    # Ensure no NaN/Inf values propagated through the model
    if np.isnan(val_preds).any() or np.isinf(val_preds).any():
        raise ValueError("XGBoost produced invalid values (NaN/Inf) in validation predictions.")
    if np.isnan(test_preds).any() or np.isinf(test_preds).any():
        raise ValueError("XGBoost produced invalid values (NaN/Inf) in test predictions.")

    print(f"Training complete. Best iteration: {model.best_iteration}")
    print(f"Validation probabilities shape: {val_preds.shape}")
    print(f"Test probabilities shape: {test_preds.shape}")

    return val_preds, test_preds


# ===== Model Registry =====
# Register the XGBoost GPU engine for the pipeline
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "xgb_gpu": train_xgb_gpu,
}
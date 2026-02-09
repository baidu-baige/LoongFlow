import cudf
import cuml
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple, Dict, Callable, Any, List
from cuml.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from scipy.special import softmax

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/leaf-classification/prepared/public"
OUTPUT_DATA_PATH = "output/5e63fe40-52af-4d8b-ac71-4d3a91b9999f/54/executor/output"

# Task-adaptive type definitions
X = cudf.DataFrame
y = cudf.Series
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

def align_to_total_classes(probs: np.ndarray, model_classes: np.ndarray, total_classes: int = 99) -> np.ndarray:
    """
    Ensures that the probability matrix has exactly 'total_classes' columns, 
    mapping the model's local class indices to the global label space.
    """
    aligned = np.zeros((probs.shape[0], total_classes), dtype=np.float32)
    for i, cls_idx in enumerate(model_classes):
        idx = int(cls_idx)
        if idx < total_classes:
            aligned[:, idx] = probs[:, i]
    return aligned

def finalize_probs(probs: np.ndarray) -> np.ndarray:
    """
    Applies row-normalization and clipping to ensure valid log-loss evaluation.
    """
    eps = 1e-15
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums
    probs = np.clip(probs, eps, 1 - eps)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs

# ===== Training Functions =====

def train_calibrated_heterogeneous_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a 7-model heterogeneous committee with intra-fold SLSQP weight calibration (L2=0.01).
    
    Models:
    1. LogisticRegression (cuml): C=0.01, penalty='l2'
    2. SVC (sklearn): C=0.1, kernel='rbf', probability=True
    3. XGBClassifier (xgboost): n_estimators=50, learning_rate=0.05, max_depth=3, label_smoothing=0.1
    4. ExtraTreesClassifier (sklearn): n_estimators=500, max_depth=10, min_samples_leaf=5
    5. MLPClassifier (sklearn): hidden_layer_sizes=(256, 128), alpha=15.0, early_stopping=True
    6. CatBoostClassifier: iterations=500, depth=4, task_type='GPU', loss_function='MultiClass'
    7. KernelRidge (sklearn): alpha=1.0, kernel='polynomial' (Softmax mapped)
    """
    print("Training Calibrated Heterogeneous Ensemble (7 models): Initializing committee...")
    
    total_n_classes = 99
    cuml.set_global_output_type('numpy')
    
    # Data conversion for CPU-based and hybrid models
    y_train_np = y_train.to_numpy().astype('int32')
    y_val_np = y_val.to_numpy().astype('int32')
    X_train_host = X_train.to_pandas()
    X_val_host = X_val.to_pandas()
    X_test_host = X_test.to_pandas()
    
    # Local label encoding to handle fold-specific class subsets
    le = SkLabelEncoder()
    y_train_local = le.fit_transform(y_train_np)
    fold_global_classes = le.classes_
    
    # OOF and Test Prediction containers
    val_signals = []
    test_signals = []

    # 1. Logistic Regression (GPU)
    print("Ensemble Model 1/7: LogisticRegression...")
    lr = LogisticRegression(C=0.01, penalty='l2', solver='qn', max_iter=2000)
    lr.fit(X_train, y_train_local)
    val_signals.append(align_to_total_classes(lr.predict_proba(X_val), fold_global_classes[lr.classes_], total_n_classes))
    test_signals.append(align_to_total_classes(lr.predict_proba(X_test), fold_global_classes[lr.classes_], total_n_classes))

    # 2. SVC (CPU)
    print("Ensemble Model 2/7: SVC...")
    svc = SVC(C=0.1, kernel='rbf', probability=True, random_state=42)
    svc.fit(X_train_host, y_train_local)
    val_signals.append(align_to_total_classes(svc.predict_proba(X_val_host), fold_global_classes[svc.classes_], total_n_classes))
    test_signals.append(align_to_total_classes(svc.predict_proba(X_test_host), fold_global_classes[svc.classes_], total_n_classes))

    # 3. XGBoost (GPU)
    print("Ensemble Model 3/7: XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        label_smoothing=0.1,
        tree_method='hist',
        device='cuda',
        random_state=42
    )
    xgb_model.fit(X_train, y_train_local)
    val_signals.append(align_to_total_classes(xgb_model.predict_proba(X_val), fold_global_classes[xgb_model.classes_], total_n_classes))
    test_signals.append(align_to_total_classes(xgb_model.predict_proba(X_test), fold_global_classes[xgb_model.classes_], total_n_classes))

    # 4. Extra Trees (CPU)
    print("Ensemble Model 4/7: ExtraTrees...")
    et = ExtraTreesClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
    et.fit(X_train_host, y_train_local)
    val_signals.append(align_to_total_classes(et.predict_proba(X_val_host), fold_global_classes[et.classes_], total_n_classes))
    test_signals.append(align_to_total_classes(et.predict_proba(X_test_host), fold_global_classes[et.classes_], total_n_classes))

    # 5. MLP (CPU)
    print("Ensemble Model 5/7: MLP (256, 128)...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=15.0, early_stopping=True, max_iter=2000, random_state=42)
    mlp.fit(X_train_host, y_train_local)
    val_signals.append(align_to_total_classes(mlp.predict_proba(X_val_host), fold_global_classes[mlp.classes_], total_n_classes))
    test_signals.append(align_to_total_classes(mlp.predict_proba(X_test_host), fold_global_classes[mlp.classes_], total_n_classes))

    # 6. CatBoost (GPU)
    print("Ensemble Model 6/7: CatBoost...")
    cb = CatBoostClassifier(
        iterations=500,
        depth=4,
        task_type='GPU',
        loss_function='MultiClass',
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    cb.fit(X_train_host, y_train_local)
    val_signals.append(align_to_total_classes(cb.predict_proba(X_val_host), fold_global_classes[cb.classes_], total_n_classes))
    test_signals.append(align_to_total_classes(cb.predict_proba(X_test_host), fold_global_classes[cb.classes_], total_n_classes))

    # 7. Kernel Ridge (CPU) - Softmax mapped
    print("Ensemble Model 7/7: KernelRidge (Polynomial)...")
    kr = KernelRidge(alpha=1.0, kernel='polynomial', degree=3)
    ohe = OneHotEncoder(sparse_output=False)
    y_train_ohe = ohe.fit_transform(y_train_local.reshape(-1, 1))
    kr.fit(X_train_host, y_train_ohe)
    
    kr_val_raw = kr.predict(X_val_host)
    kr_test_raw = kr.predict(X_test_host)
    kr_val_probs = softmax(kr_val_raw, axis=1)
    kr_test_probs = softmax(kr_test_raw, axis=1)
    kr_classes = ohe.categories_[0].astype(int)
    val_signals.append(align_to_total_classes(kr_val_probs, fold_global_classes[kr_classes], total_n_classes))
    test_signals.append(align_to_total_classes(kr_test_probs, fold_global_classes[kr_classes], total_n_classes))

    # --- Intra-Fold SLSQP Calibration with L2 Regularization ---
    print("Optimizing 7-model ensemble weights via SLSQP (L2=0.01)...")
    
    def objective(weights):
        # Weighted sum of probabilities
        blend = np.zeros_like(val_signals[0])
        for w, s in zip(weights, val_signals):
            blend += w * s
        blend = finalize_probs(blend)
        # Multi-class Log Loss + L2 Penalty for weight regularization
        loss_val = log_loss(y_val_np, blend, labels=np.arange(total_n_classes))
        penalty = 0.01 * np.sum(np.square(weights))
        return loss_val + penalty

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, 1) for _ in range(len(val_signals))]
    init_weights = np.ones(len(val_signals)) / len(val_signals)
    
    res = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = res.x
    print(f"Optimal 7-model weights: {best_weights}")

    # Final Blending
    val_preds = np.zeros_like(val_signals[0])
    test_preds = np.zeros_like(test_signals[0])
    for w, v_s, t_s in zip(best_weights, val_signals, test_signals):
        val_preds += w * v_s
        test_preds += w * t_s

    val_preds = finalize_probs(val_preds)
    test_preds = finalize_probs(test_preds)

    print("Heterogeneous ensemble calibration complete.")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "calibrated_heterogeneous_ensemble": train_calibrated_heterogeneous_ensemble,
}
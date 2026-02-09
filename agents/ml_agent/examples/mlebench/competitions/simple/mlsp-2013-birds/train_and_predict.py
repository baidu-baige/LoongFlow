import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.multiclass import OneVsRestClassifier
from typing import Tuple, Dict, Callable, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-10/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/9f7a14b2-9e2e-4beb-a8af-238199431c62/57/executor/output"

# Task-adaptive type definitions
X = np.ndarray           # Feature matrix (N, 250)
y = np.ndarray           # Target matrix (N, 19)
Predictions = np.ndarray # Probability matrix (N, 19)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Training Functions =====

def train_per_species_pseudo_triple_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a Per-Species Pseudo-Label Augmented Triple-GBDT Ensemble.
    
    Stage A: Train base Triple-GBDT Ensemble on labeled data.
    Stage B: For each species, identify high-confidence test samples (Prob > 0.90 or Prob < 0.01).
    Stage C: Augment training data and re-train the final ensemble per species.
    
    Hardware Optimization & Compatibility:
    - XGBoost: GPU (cuda)
    - LightGBM: CPU (CUDA build unavailable in environment)
    - CatBoost: CPU (rsm parameter not supported on GPU for Logloss)
    """
    num_species = 19
    print(f"Stage 4: Initializing Per-Species Pseudo-Label Augmented Ensemble (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})")

    # Parameters as per technical specification
    xgb_params = {
        'n_estimators': 150,
        'max_depth': 3,
        'learning_rate': 0.03,
        'colsample_bytree': 0.4,
        'min_child_weight': 2,
        'random_state': 42,
        'device': 'cuda',
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'verbosity': 0,
        'n_jobs': 1
    }
    
    lgbm_params = {
        'n_estimators': 150,
        'num_leaves': 7,
        'learning_rate': 0.03,
        'colsample_bytree': 0.4,
        'random_state': 42,
        'device': 'cpu',
        'verbose': -1,
        'n_jobs': -1
    }
    
    cat_params = {
        'iterations': 150,
        'depth': 3,
        'learning_rate': 0.03,
        'rsm': 0.4,
        'loss_function': 'Logloss',
        'task_type': 'CPU',
        'random_state': 42,
        'verbose': 0,
        'thread_count': -1,
        'allow_writing_files': False
    }

    # --- Stage A: Train base models on labeled data ---
    print("Stage A: Training base ensemble on labeled data...")
    # Use OneVsRest parallelization for CPU models, sequential for GPU XGB to avoid context overhead
    ovr_xgb = OneVsRestClassifier(xgb.XGBClassifier(**xgb_params), n_jobs=1)
    ovr_lgbm = OneVsRestClassifier(lgb.LGBMClassifier(**lgbm_params), n_jobs=-1)
    ovr_cat = OneVsRestClassifier(cb.CatBoostClassifier(**cat_params), n_jobs=-1)

    ovr_xgb.fit(X_train, y_train)
    ovr_lgbm.fit(X_train, y_train)
    ovr_cat.fit(X_train, y_train)

    # Generate initial test predictions for per-species pseudo-labeling
    xgb_test_a = ovr_xgb.predict_proba(X_test).astype(np.float32)
    lgbm_test_a = ovr_lgbm.predict_proba(X_test).astype(np.float32)
    cat_test_a = ovr_cat.predict_proba(X_test).astype(np.float32)
    
    # Base ensemble blend: 0.4 XGB + 0.3 LGBM + 0.3 Cat
    base_test_preds = (0.4 * xgb_test_a + 0.3 * lgbm_test_a + 0.3 * cat_test_a)

    # --- Stage B/C: Per-species augmentation and re-training ---
    print("Stage B/C: Re-training models with high-confidence pseudo-labels per species...")
    final_val_preds = np.zeros((X_val.shape[0], num_species), dtype=np.float32)
    final_test_preds = np.zeros((X_test.shape[0], num_species), dtype=np.float32)

    for s in range(num_species):
        if s % 5 == 0:
            print(f"Refining species {s}/{num_species}...")
            
        # Select high-confidence samples for species s from X_test
        pos_indices = np.where(base_test_preds[:, s] > 0.90)[0]
        neg_indices = np.where(base_test_preds[:, s] < 0.01)[0]
        
        # Prepare augmented training set for this label
        if len(pos_indices) > 0 or len(neg_indices) > 0:
            X_pseudo = X_test[np.concatenate([pos_indices, neg_indices])]
            y_pseudo = np.concatenate([np.ones(len(pos_indices)), np.zeros(len(neg_indices))])
            
            X_aug = np.concatenate([X_train, X_pseudo])
            y_aug = np.concatenate([y_train[:, s], y_pseudo])
        else:
            X_aug = X_train
            y_aug = y_train[:, s]
        
        # Binary target check
        unique_y = np.unique(y_aug)
        if len(unique_y) < 2:
            constant_val = float(unique_y[0])
            final_val_preds[:, s] = constant_val
            final_test_preds[:, s] = constant_val
            continue

        # Re-train models on augmented data
        final_xgb = xgb.XGBClassifier(**xgb_params)
        final_lgbm = lgb.LGBMClassifier(**lgbm_params)
        final_cat = cb.CatBoostClassifier(**cat_params)
        
        final_xgb.fit(X_aug, y_aug)
        final_lgbm.fit(X_aug, y_aug)
        final_cat.fit(X_aug, y_aug)
        
        # Predict on Val and Test
        p_xgb_val = final_xgb.predict_proba(X_val)[:, 1]
        p_lgbm_val = final_lgbm.predict_proba(X_val)[:, 1]
        p_cat_val = final_cat.predict_proba(X_val)[:, 1]
        
        p_xgb_test = final_xgb.predict_proba(X_test)[:, 1]
        p_lgbm_test = final_lgbm.predict_proba(X_test)[:, 1]
        p_cat_test = final_cat.predict_proba(X_test)[:, 1]
        
        # Final Weighted Blend for current species
        final_val_preds[:, s] = (0.4 * p_xgb_val + 0.3 * p_lgbm_val + 0.3 * p_cat_val)
        final_test_preds[:, s] = (0.4 * p_xgb_test + 0.3 * p_lgbm_test + 0.3 * p_cat_test)

    # Final cleanup and sanity checks
    final_val_preds = np.nan_to_num(np.clip(final_val_preds, 0.0, 1.0))
    final_test_preds = np.nan_to_num(np.clip(final_test_preds, 0.0, 1.0))

    print(f"Augmented ensemble training complete. Mean test probability: {np.mean(final_test_preds):.4f}")
    return final_val_preds, final_test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "per_species_pseudo_triple_ensemble": train_per_species_pseudo_triple_ensemble,
}
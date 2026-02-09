import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Dict

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Path constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/3c8d5cca-ccb7-4c25-92f1-7d0f571dedc1/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Ensures data robustness, performs feature selection, and executes cross-validated training.
    """
    print("Initializing production pipeline for NOMAD2018 task...")
    
    # 1. Load full dataset
    # Requirement: MUST call load_data(validation_mode=False)
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)
    
    # 2. Data Robustness Fix
    # Address the 'mat1 and mat2 shapes cannot be multiplied (1x0 and 3x3)' error
    # This occurs in preprocess.py when atom_coords has an invalid shape.
    def fix_atom_coords(x):
        arr = np.asanyarray(x)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        try:
            # Attempt to reshape any flattened or empty arrays into (N, 3)
            return arr.reshape(-1, 3).astype(np.float32)
        except Exception:
            # Fallback to empty structure if data is fundamentally malformed
            return np.zeros((0, 3), dtype=np.float32)

    X_train_raw['atom_coords'] = X_train_raw['atom_coords'].apply(fix_atom_coords)
    X_test_raw['atom_coords'] = X_test_raw['atom_coords'].apply(fix_atom_coords)
    
    # 3. Set up splitting strategy
    splitter = get_splitter(X_train_raw, y_train_raw)
    n_splits = splitter.get_n_splits()
    
    # Storage for Cross-Validation artifacts
    model_name = "gbdt_ensemble"
    train_fn = PREDICTION_ENGINES[model_name]
    oof_preds = np.zeros((len(X_train_raw), 2))
    test_preds_list = []
    
    # 4. Cross-Validation execution
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw, y_train_raw)):
        print(f"\n--- Processing Fold {fold_idx + 1}/{n_splits} ---")
        
        # Partition data
        X_tr, X_va = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
        y_tr, y_va = y_train_raw.iloc[train_idx], y_train_raw.iloc[val_idx]
        
        # Pipeline: Preprocessing
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr, y_tr, X_va, y_va, X_test_raw
        )
        
        # Pipeline: Feature Selection (Technical Specification Requirement #3)
        # Use a secondary model to identify and prune zero-importance features
        print("  Evaluating feature importance for pruning...")
        rf_selector = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        # Target for selection: Mean log-transformed target values
        y_target_log = np.log1p(np.maximum(0, y_tr_p.values)).mean(axis=1)
        rf_selector.fit(X_tr_p, y_target_log)
        
        # Filter features with zero or near-zero importance to reduce noise
        importance_threshold = 1e-7
        keep_cols = X_tr_p.columns[rf_selector.feature_importances_ > importance_threshold]
        print(f"  Feature Selection: {len(X_tr_p.columns)} -> {len(keep_cols)} features retained.")
        
        X_tr_p = X_tr_p[keep_cols]
        X_va_p = X_va_p[keep_cols]
        X_te_p = X_te_p[keep_cols]
        
        # Pipeline: Training and Prediction
        val_p, test_p = train_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
        
        # Store results
        oof_preds[val_idx] = val_p
        test_preds_list.append(test_p)
        
    # 5. Aggregate predictions
    # Calculate simple average of test predictions across folds
    avg_test_preds = np.mean(test_preds_list, axis=0)
    
    # 6. Ensemble Module (Stage 5)
    # Merges predictions in log-space using performance-based weighting
    all_val_preds = {model_name: oof_preds}
    all_test_preds = {model_name: avg_test_preds}
    final_test_preds = ensemble(all_val_preds, all_test_preds, y_train_raw)
    
    # 7. Deliverable Generation
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    
    submission_df = pd.DataFrame({
        'id': test_ids,
        'formation_energy_ev_natom': final_test_preds[:, 0],
        'bandgap_energy_ev': final_test_preds[:, 1]
    })
    submission_df.to_csv(submission_file_path, index=False)
    
    # 8. Execution Summary
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": {
            "oof": {
                "mean": float(np.mean(oof_preds)),
                "std": float(np.std(oof_preds)),
                "min": float(np.min(oof_preds)),
                "max": float(np.max(oof_preds)),
            },
            "test": {
                "mean": float(np.mean(final_test_preds)),
                "std": float(np.std(final_test_preds)),
                "min": float(np.min(final_test_preds)),
                "max": float(np.max(final_test_preds)),
            }
        },
    }
    
    print(f"\nPipeline execution complete. Submission saved to: {submission_file_path}")
    return output_info
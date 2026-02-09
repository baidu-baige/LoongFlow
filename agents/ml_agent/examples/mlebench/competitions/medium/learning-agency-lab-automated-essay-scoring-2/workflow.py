import os
import gc
import cudf
import numpy as np
import pandas as pd
import torch
import cupy as cp
from typing import Dict

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Global Paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/learning-agency-lab-automated-essay-scoring-2/prepared/public"
OUTPUT_DATA_PATH = "output/60558907-7467-4c6f-a8d6-3b62cb9514db/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This function implements a 5-fold cross-validation strategy, training a blended 
    DeBERTa-v3 and LightGBM model per fold, and finalizes with threshold optimization 
    to maximize the Quadratic Weighted Kappa.
    """
    print("Stage 6: Workflow starting.")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    # We use validation_mode=False to ensure the entire training set is used.
    print("Step 1: Loading full dataset...")
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    # StratifiedKFold (cuml) maintains the score distribution (1-6) across folds on GPU.
    print("Step 2: Initializing splitter...")
    splitter = get_splitter(X_train_raw, y_train_raw)
    n_splits = splitter.get_n_splits()

    # Placeholders for Out-of-Fold (OOF) and Test predictions
    # We store raw regression outputs (NumPy) and optimize thresholds at the end.
    oof_preds = np.zeros(len(y_train_raw), dtype=np.float32)
    test_preds_accum = np.zeros(len(X_test_raw), dtype=np.float32)

    # 3. Sequential Cross-Validation Loop
    print(f"Step 3: Starting {n_splits}-fold training loop...")
    engine = PREDICTION_ENGINES["deberta_lgbm_regressor"]
    
    # cuml splitters return indices as CuPy arrays or GPU-backed sequences
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw, y_train_raw)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")
        
        # Split data for this fold using GPU-accelerated indexing
        X_tr = X_train_raw.iloc[train_idx]
        y_tr = y_train_raw.iloc[train_idx]
        X_vl = X_train_raw.iloc[val_idx]
        y_vl = y_train_raw.iloc[val_idx]

        # a. Preprocess: Generate dual-stream features (Linguistic + Transformer tokens)
        print(f"Fold {fold + 1}: Preprocessing...")
        X_tr_p, y_tr_p, X_vl_p, y_vl_p, X_te_p = preprocess(X_tr, y_tr, X_vl, y_vl, X_test_raw)

        # b. Train & Predict: Execute DeBERTa (DDP) and LightGBM (GPU)
        print(f"Fold {fold + 1}: Training models...")
        val_p, test_p = engine(X_tr_p, y_tr_p, X_vl_p, y_vl_p, X_te_p)

        # c. Accumulate results
        # Conversion: Must explicitly convert GPU indices (CuPy/cuDF) to NumPy for indexing a NumPy array
        if hasattr(val_idx, 'get'):
            idx_np = val_idx.get()
        elif hasattr(val_idx, 'to_numpy'):
            idx_np = val_idx.to_numpy()
        else:
            idx_np = cp.asnumpy(val_idx)

        oof_preds[idx_np] = val_p
        test_preds_accum += test_p / n_splits

        # Resource Management: Clear fold-specific artifacts
        del X_tr, y_tr, X_vl, y_vl, X_tr_p, y_tr_p, X_vl_p, y_vl_p, X_te_p, val_p, test_p
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Final Ensemble & Threshold Optimization
    print("\nStep 4: Optimizing thresholds and ensembling...")
    # Convert target to numpy for the ensemble/optimization stage
    y_train_np = y_train_raw.to_pandas().values if hasattr(y_train_raw, 'to_pandas') else np.array(y_train_raw)
    
    # Use the optimized ensemble function from Stage 5
    final_test_labels = ensemble(
        all_val_preds={"blended_cv": oof_preds},
        all_test_preds={"blended_cv": test_preds_accum},
        y_val=y_train_np
    )

    # 5. Compute Prediction Statistics
    print("Step 5: Computing statistics...")
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_preds)),
            "std": float(np.std(oof_preds)),
            "min": float(np.min(oof_preds)),
            "max": float(np.max(oof_preds)),
        },
        "test": {
            "mean": float(np.mean(final_test_labels)),
            "std": float(np.std(final_test_labels)),
            "min": float(np.min(final_test_labels)),
            "max": float(np.max(final_test_labels)),
        }
    }

    # 6. Generate Deliverables
    print("Step 6: Generating submission file...")
    test_ids_np = test_ids.to_pandas().values if hasattr(test_ids, 'to_pandas') else np.array(test_ids)
    submission_df = pd.DataFrame({
        'essay_id': test_ids_np,
        'score': final_test_labels
    })
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    # Save OOF predictions for audit trail
    np.save(os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy"), oof_preds)

    print(f"Workflow complete. Submission saved to {submission_file_path}")
    
    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "oof_file_path": os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy")
    }
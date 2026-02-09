import os
import pandas as pd
import numpy as np
from typing import Dict

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/6795de84-a7ab-443d-bbf5-03771db15966/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting pipeline orchestration...")
    
    # 0. Ensure output directory exists
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    # Returns raw waveforms [N, 4000] and labels
    X_train_raw, y_train_raw, X_test_raw, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # Configured for 5-fold stratified validation
    splitter = get_splitter(X_train_raw, y_train_raw)
    
    # Containers for Cross-Validation results
    oof_preds = np.zeros(len(y_train_raw), dtype=np.float32)
    all_test_preds_dict = {}
    
    # 3. Execute Fold Loop
    print(f"Starting cross-validation with {splitter.get_n_splits()} folds.")
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_raw, y_train_raw)):
        print(f"--- Processing Fold {fold_idx} ---")
        
        # Split data for this fold
        X_tr_fold, X_va_fold = X_train_raw[train_idx], X_train_raw[val_idx]
        y_tr_fold, y_va_fold = y_train_raw[train_idx], y_train_raw[val_idx]
        
        # a. Preprocess: Raw Waveforms -> PCEN Mel Spectrograms (N, 128, 63, 1)
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr_fold, y_tr_fold, X_va_fold, y_va_fold, X_test_raw
        )
        
        # b. Train and Predict using EfficientNet-B0
        # Engine utilizes GPU DDP for training
        train_fn = PREDICTION_ENGINES["efficientnet_b0"]
        val_preds, test_preds = train_fn(
            X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        )
        
        # c. Record results
        oof_preds[val_idx] = val_preds
        # Store test predictions from each fold for rank averaging ensemble
        all_test_preds_dict[f"efficientnet_b0_fold_{fold_idx}"] = test_preds
        
        # Clean up temporary fold variables to manage memory efficiently
        del X_tr_p, X_va_p, X_te_p, val_preds, test_preds

    # 4. Ensemble stage
    # Prepare inputs for the rank-averaging ensemble
    # We pass the full OOF for performance reporting
    all_val_preds_dict = {"efficientnet_b0_oof": oof_preds}
    
    final_test_scores = ensemble(
        all_val_preds=all_val_preds_dict,
        all_test_preds=all_test_preds_dict,
        y_val=y_train_raw
    )

    # 5. Compute deliverables and statistics
    # Generate submission file
    submission_df = pd.DataFrame({
        "clip": test_ids,
        "probability": final_test_scores
    })
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")

    # Compute statistics for return
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_preds)),
            "std": float(np.std(oof_preds)),
            "min": float(np.min(oof_preds)),
            "max": float(np.max(oof_preds)),
        },
        "test": {
            "mean": float(np.mean(final_test_scores)),
            "std": float(np.std(final_test_scores)),
            "min": float(np.min(final_test_scores)),
            "max": float(np.max(final_test_scores)),
        }
    }

    # Save OOF predictions for audit/future ensembling
    oof_path = os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy")
    np.save(oof_path, oof_preds)

    output_info = {
        "submission_file_path": submission_file_path,
        "oof_predictions_path": oof_path,
        "prediction_stats": prediction_stats,
    }
    
    print("Pipeline execution completed successfully.")
    return output_info

if __name__ == "__main__":
    workflow()
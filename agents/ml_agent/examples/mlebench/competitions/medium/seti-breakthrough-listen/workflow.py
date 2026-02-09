import os
import gc
import torch
import numpy as np
import pandas as pd
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/seti-breakthrough-listen/prepared/public"
OUTPUT_DATA_PATH = "output/a429c40e-fe12-455c-8b05-ca9d732aabeb/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline for the SETI Breakthrough Listen task.
    This executes a 5-fold cross-validation strategy using EfficientNet-B5 and generates the final submission.
    """
    print("Starting production workflow...")

    # 1. Load the COMPLETE dataset
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)
    
    # 2. Set up the data splitting strategy (StratifiedKFold)
    splitter = get_splitter(X_train_full, y_train_full)
    n_splits = splitter.get_n_splits()
    
    # 3. Initialize containers for Out-of-Fold (OOF) and Test predictions
    # oof_preds will store validation predictions for the entire training set
    oof_preds = np.zeros(len(y_train_full))
    # all_test_preds will store test set predictions from each fold to be ensembled
    all_test_preds = {}
    
    # 4. Execute the K-Fold Cross-Validation loop
    train_fn = PREDICTION_ENGINES["efficientnet_b5"]
    
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")
        
        # Split the raw metadata DataFrames for the current fold
        X_tr_fold = X_train_full.iloc[train_idx]
        y_tr_fold = y_train_full.iloc[train_idx]
        X_va_fold = X_train_full.iloc[val_idx]
        y_va_fold = y_train_full.iloc[val_idx]
        
        # Stage: Preprocess
        # Transforms raw .npy files into stacked spectrogram tensors
        print(f"Folding {fold + 1}: Preprocessing data...")
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr_fold, y_tr_fold, X_va_fold, y_va_fold, X_test_full
        )
        
        # Stage: Train and Predict
        # Executes distributed training on available GPUs
        print(f"Folding {fold + 1}: Training model and generating predictions...")
        val_p, test_p = train_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
        
        # Store fold results
        oof_preds[val_idx] = val_p
        all_test_preds[f"fold_{fold}"] = test_p
        
        # Efficiency: Explicit memory management to free up RAM and VRAM for the next fold
        print(f"Folding {fold + 1}: Cleaning up resources...")
        del X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        gc.collect()
        torch.cuda.empty_cache()
    
    # 5. Stage: Ensemble
    # Aggregates test predictions from all folds and evaluates OOF performance
    print("\nStarting final ensemble and evaluation...")
    # Wrap OOF in a dict to satisfy ensemble function signature requirements
    all_val_preds = {"oof_ensemble": oof_preds}
    final_test_predictions = ensemble(
        all_val_preds=all_val_preds,
        all_test_preds=all_test_preds,
        y_val=y_train_full.values
    )
    
    # 6. Compute Prediction Statistics
    # Ensuring all values are converted to standard Python floats for JSON serialization
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_preds)),
            "std": float(np.std(oof_preds)),
            "min": float(np.min(oof_preds)),
            "max": float(np.max(oof_preds)),
        },
        "test": {
            "mean": float(np.mean(final_test_predictions)),
            "std": float(np.std(final_test_predictions)),
            "min": float(np.min(final_test_predictions)),
            "max": float(np.max(final_test_predictions)),
        }
    }
    
    # 7. Generate and Save Deliverables
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
        
    # Create submission file
    submission_df = pd.DataFrame({
        "id": test_ids,
        "target": final_test_predictions
    })
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    # Save OOF predictions as a permanent artifact
    oof_save_path = os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy")
    np.save(oof_save_path, oof_preds)
    
    print(f"Workflow execution complete.")
    print(f"Final submission saved to: {submission_file_path}")
    print(f"OOF artifacts saved to: {oof_save_path}")

    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "oof_artifact_path": oof_save_path
    }
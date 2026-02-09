import os
import numpy as np
import pandas as pd
from typing import Dict
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/cf83edc4-8764-4cf8-95a0-4f4a823260c7/2/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Executes a 5-fold cross-validation strategy using a Dual Backbone (ConvNeXt + Swin) 
    ensemble with TTA, followed by rank-based ensembling to produce the final submission.
    """
    print("Starting production workflow...")

    # 1. Load full dataset
    # validation_mode=False ensures we use all available training and test data.
    X_train_meta, y_train, X_test_meta, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    # Stratified 5-Fold CV ensures robust estimation of performance.
    splitter = get_splitter(X_train_meta, y_train)
    
    # Initialize containers for Out-of-Fold (OOF) and Test predictions
    oof_preds = np.zeros(len(X_train_meta))
    test_preds_accum = np.zeros(len(X_test_meta))
    
    # 3. Sequential Cross-Validation Loop
    print(f"Beginning 5-fold cross-validation loop...")
    fold = 0
    for train_idx, val_idx in splitter.split(X_train_meta, y_train):
        fold += 1
        print(f"--- Processing Fold {fold}/5 ---")
        
        # Split metadata
        X_tr_meta, X_vl_meta = X_train_meta.iloc[train_idx], X_train_meta.iloc[val_idx]
        y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # a. Preprocess: Load images into RAM and setup augmentation pipelines
        # This function handles the D4 symmetry group requirement.
        X_tr_p, y_tr_p, X_vl_p, y_vl_p, X_ts_p = preprocess(
            X_tr_meta, y_tr, X_vl_meta, y_vl, X_test_meta
        )
        
        # b. Train and Predict: Dual Backbone Ensemble (ConvNeXt-S + Swin-S)
        # This function leverages all 4 GPUs via Distributed Data Parallel (DDP).
        engine = PREDICTION_ENGINES["dual_backbone_ensemble"]
        val_p, test_p = engine(X_tr_p, y_tr_p, X_vl_p, y_vl_p, X_ts_p)
        
        # Store results
        oof_preds[val_idx] = val_p
        test_preds_accum += test_p / 5.0 # Simple averaging across folds before final ensembling
        
        print(f"Fold {fold} complete.")

    # 4. Ensemble stage
    # Use Weighted Rank Averaging to finalize predictions. 
    # Although we use one core model engine, the ensemble function applies rank-normalization 
    # which is optimal for AUC-ROC evaluation.
    print("Finalizing predictions via ensembling...")
    all_val_preds = {"dual_backbone_cv": oof_preds}
    all_test_preds = {"dual_backbone_cv": test_preds_accum}
    
    final_test_preds = ensemble(all_val_preds, all_test_preds, y_train)

    # 5. Compute deliverables and statistics
    # Ensure all values are JSON-serializable (native Python types)
    prediction_stats = {
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
    }

    # 6. Generate and save submission file
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_df = pd.DataFrame({
        'id': test_ids.values,
        'label': final_test_preds
    })
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")

    # Save OOF predictions for potential diagnostic use
    oof_file_path = os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy")
    np.save(oof_file_path, oof_preds)

    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "oof_file_path": oof_file_path
    }
    
    print("Workflow execution completed successfully.")
    return output_info

if __name__ == "__main__":
    workflow()
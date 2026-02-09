import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/petfinder-pawpularity-score/prepared/public"
OUTPUT_DATA_PATH = "output/10daf1c7-eb8a-49b6-bcf3-ba43767dcbe6/2/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    Executes a 5-fold cross-validation strategy, training a hybrid ensemble of 
    ConvNeXt-Large and Swin-Large on each fold, and generating final test predictions.
    """
    print("Initializing production workflow...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy (5-fold stratified)
    splitter = get_splitter(X_train_full, y_train_full)
    n_splits = splitter.get_n_splits()
    
    # Initialize containers for Out-of-Fold and Test predictions
    # train_and_predict returns values in range [0, 100]
    oof_predictions = np.zeros(len(X_train_full))
    test_predictions_accumulator = np.zeros(len(X_test_full))
    
    # Use the hybrid engine defined in the specification
    engine_name = "hybrid_convnext_swin"
    engine = PREDICTION_ENGINES[engine_name]
    
    # 3. Execution of the 5-fold cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n{'='*20} Starting Fold {fold + 1}/{n_splits} {'='*20}")
        
        # Split train/validation data for this fold
        X_train_fold = X_train_full.iloc[train_idx].reset_index(drop=True)
        y_train_fold = y_train_full.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X_train_full.iloc[val_idx].reset_index(drop=True)
        y_val_fold = y_train_full.iloc[val_idx].reset_index(drop=True)
        
        # a. Preprocess fold data into PyTorch Datasets
        X_train_ds, _, X_val_ds, _, X_test_ds = preprocess(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_full
        )
        
        # b. Train model engine and generate predictions
        # val_preds and test_preds are (N, 2) arrays: [ConvNeXt, Swin]
        val_preds_fold, test_preds_fold = engine(
            X_train_ds, y_train_fold, X_val_ds, y_val_fold, X_test_ds
        )
        
        # c. Apply ensemble weights (0.6 ConvNeXt + 0.4 Swin) for both OOF and Test
        # We use the provided ensemble function for test predictions
        all_val_dict = {engine_name: val_preds_fold}
        all_test_dict = {engine_name: test_preds_fold}
        
        # The ensemble function returns the weighted test ensemble for this fold
        fold_test_ensemble = ensemble(all_val_dict, all_test_dict, y_val_fold)
        test_predictions_accumulator += fold_test_ensemble / n_splits
        
        # For OOF, we manually calculate the weighted average as ensemble() only returns test
        # index 0: ConvNeXt, index 1: Swin
        fold_oof_ensemble = (0.6 * val_preds_fold[:, 0]) + (0.4 * val_preds_fold[:, 1])
        oof_predictions[val_idx] = fold_oof_ensemble
        
        print(f"Fold {fold + 1} completed.")

    # 4. Final aggregation and artifact generation
    print("\nProcessing final results...")
    
    # Final test predictions (already averaged over 5 folds)
    # Ensure they are clipped to valid Pawpularity range [1, 100]
    final_test_predictions = np.clip(test_predictions_accumulator, 1.0, 100.0)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'Pawpularity': final_test_predictions
    })
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")
    
    # Save OOF predictions for downstream analysis or meta-modeling
    oof_file_path = os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy")
    np.save(oof_file_path, oof_predictions)

    # 5. Compute and format prediction statistics
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_predictions)),
            "std": float(np.std(oof_predictions)),
            "min": float(np.min(oof_predictions)),
            "max": float(np.max(oof_predictions)),
        },
        "test": {
            "mean": float(np.mean(final_test_predictions)),
            "std": float(np.std(final_test_predictions)),
            "min": float(np.min(final_test_predictions)),
            "max": float(np.max(final_test_predictions)),
        }
    }
    
    print("Workflow execution complete.")
    
    return {
        "submission_file_path": submission_file_path,
        "oof_file_path": oof_file_path,
        "prediction_stats": prediction_stats
    }

if __name__ == "__main__":
    results = workflow()
    print(json.dumps(results, indent=4))
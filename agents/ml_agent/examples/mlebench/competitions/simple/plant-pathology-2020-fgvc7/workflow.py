import os
import pandas as pd
import numpy as np
import torch
from typing import Dict

# Import component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/fc83b0b0-0bd1-41b2-8cd2-ac42c45cd457/2/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Executes a 5-fold Stratified Cross-Validation using an ensemble of CNN backbones.
    """
    print("Starting production workflow...")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    # Returns image paths, multi-label targets, test paths, and test IDs
    X_train_paths, y_train, X_test_paths, test_ids = load_data(validation_mode=False)
    
    # 2. Define splitting strategy (5-fold Stratified CV)
    splitter = get_splitter(X_train_paths, y_train)
    n_splits = splitter.get_n_splits()
    
    # Initialize containers for out-of-fold (OOF) and test predictions
    num_samples = len(X_train_paths)
    num_classes = y_train.shape[1]
    oof_predictions = np.zeros((num_samples, num_classes), dtype=np.float32)
    all_test_preds_list = []

    # 3. Cross-Validation Loop
    print(f"Executing {n_splits}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_paths, y_train)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")
        
        # Split paths and labels
        X_tr_fold = [X_train_paths[i] for i in train_idx]
        y_tr_fold = y_train[train_idx]
        X_vl_fold = [X_train_paths[i] for i in val_idx]
        y_vl_fold = y_train[val_idx]
        
        # Preprocess: Convert paths to PyTorch Datasets with augmentations
        X_tr_ds, _, X_vl_ds, _, X_te_ds = preprocess(
            X_tr_fold, y_tr_fold, 
            X_vl_fold, y_vl_fold, 
            X_test_paths
        )
        
        # Train and Predict: Using the ensemble model engine (EfficientNetV2-S + ConvNeXt-Tiny)
        # This engine handles DDP training and TTA inference
        engine = PREDICTION_ENGINES["plant_pathology_ensemble"]
        fold_val_preds, fold_test_preds = engine(
            X_tr_ds, y_tr_fold,
            X_vl_ds, y_vl_fold,
            X_te_ds
        )
        
        # Store fold results
        oof_predictions[val_idx] = fold_val_preds
        all_test_preds_list.append(fold_test_preds)
        
        print(f"Fold {fold + 1} complete.")

    # 4. Aggregate and Ensemble
    # Average test predictions across all folds
    avg_test_preds = np.mean(all_test_preds_list, axis=0)
    
    # Use the ensemble function to finalize predictions and calculate OOF AUC.
    # Passing the full OOF and averaged test predictions as a single model entry.
    all_val_preds_dict = {"cv_ensemble": oof_predictions}
    all_test_preds_dict = {"cv_ensemble": avg_test_preds}
    
    final_test_predictions = ensemble(
        all_val_preds=all_val_preds_dict,
        all_test_preds=all_test_preds_dict,
        y_val=y_train
    )

    # 5. Compute Prediction Statistics
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

    # 6. Generate and Save Deliverables
    # Load original train.csv to ensure correct target column order
    train_df = pd.read_csv(os.path.join(BASE_DATA_PATH, "train.csv"))
    target_cols = [c for c in train_df.columns if c != 'image_id']
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(final_test_predictions, columns=target_cols)
    submission_df.insert(0, 'image_id', test_ids)
    
    # Save to CSV
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    print(f"Workflow execution finished. Submission saved to {submission_file_path}")

    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
    }
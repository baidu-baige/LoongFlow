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

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/chaii-hindi-and-tamil-question-answering/prepared/public"
OUTPUT_DATA_PATH = "output/af0f7d71-a062-46e3-8926-51aedd28d3b4/3/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.

    This function integrates all pipeline components (data loading, preprocessing, 
    data splitting, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    """
    print("Starting production workflow execution...")

    # 1. Load full augmented dataset
    # validation_mode=False ensures we use the complete dataset including external augmentations
    X_train_full, y_train_full, X_test_raw, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    # Stratification ensures Hindi/Tamil distribution is maintained across folds
    splitter = get_splitter(X_train_full, y_train_full)

    all_val_preds = {}
    all_test_preds = {}

    # 3. K-Fold Cross-Validation Loop
    n_splits = splitter.get_n_splits(X_train_full, y_train_full)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n--- Executing Pipeline Fold {fold + 1}/{n_splits} ---")
        
        # Split raw data for this fold
        X_train_fold = X_train_full.iloc[train_idx].reset_index(drop=True)
        y_train_fold = y_train_full.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X_train_full.iloc[val_idx].reset_index(drop=True)
        y_val_fold = y_train_full.iloc[val_idx].reset_index(drop=True)

        # Preprocess features into tokenized chunks (MuRIL + XLM-R dual path)
        print(f"Preprocessing Fold {fold + 1}...")
        X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p = preprocess(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_raw
        )

        # Train each registered model engine (MuRIL-Large and XLM-RoBERTa-Large)
        for engine_name, engine_fn in PREDICTION_ENGINES.items():
            print(f"Training model engine: {engine_name} (Fold {fold + 1})")
            
            # engine_fn handles DDP training on available GPUs and returns logits
            val_logits, test_logits = engine_fn(X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p)
            
            # Store raw logits for the ensemble stage
            key = f"{engine_name}_fold_{fold}"
            all_val_preds[key] = val_logits
            all_test_preds[key] = test_logits

    # 4. Ensemble stage
    # Aggregates windowed logits back to original sample level and performs weighted averaging
    print("\nStarting global ensemble and post-processing...")
    submission_df = ensemble(all_val_preds, all_test_preds, y_train_full)

    # 5. Verify and set submission path
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    # Ensure the file was saved by the ensemble component
    if not os.path.exists(submission_file_path):
        submission_df.to_csv(submission_file_path, index=False)
    
    print(f"Deliverable generated: {submission_file_path}")

    # 6. Compute prediction distribution statistics
    # Flatten all logits from all models and folds to compute distributions
    all_val_logits_flat = np.concatenate([v.flatten() for v in all_val_preds.values()])
    all_test_logits_flat = np.concatenate([v.flatten() for v in all_test_preds.values()])

    prediction_stats = {
        "oof": {
            "mean": float(np.mean(all_val_logits_flat)),
            "std": float(np.std(all_val_logits_flat)),
            "min": float(np.min(all_val_logits_flat)),
            "max": float(np.max(all_val_logits_flat)),
        },
        "test": {
            "mean": float(np.mean(all_test_logits_flat)),
            "std": float(np.std(all_test_logits_flat)),
            "min": float(np.min(all_test_logits_flat)),
            "max": float(np.max(all_test_logits_flat)),
        }
    }

    print("Pipeline execution complete. Summary stats computed.")
    
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "folds_processed": n_splits,
        "models_per_fold": list(PREDICTION_ENGINES.keys())
    }
    
    return output_info
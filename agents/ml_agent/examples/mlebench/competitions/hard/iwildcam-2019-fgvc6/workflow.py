import os
import numpy as np
import pandas as pd
from typing import Dict

# import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/iwildcam-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/939424a9-5a07-4c99-9f56-709187b5b05c/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    Executes a 5-fold CV strategy, ensembles results, and generates a submission file.
    """
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    print("Step 1: Loading full dataset...")
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)
    
    print("Step 2: Initializing splitter (LocationGroupSplitter)...")
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Storage for OOF and Test predictions
    oof_probs = np.zeros((len(X_train_full), 23))
    all_test_probs = {}
    
    num_folds = splitter.get_n_splits()
    print(f"Step 3: Starting {num_folds}-fold Cross-Validation...")
    
    model_fn = PREDICTION_ENGINES["convnext_base_meta"]
    
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"--- Processing Fold {fold_idx + 1}/{num_folds} ---")
        
        # Split data for this fold
        X_tr, X_va = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_va = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        # Preprocess
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr, y_tr, X_va, y_va, X_test_full
        )
        
        # Train and Predict
        # train_and_predict returns (val_probs, test_probs)
        val_probs, test_probs = model_fn(X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p)
        
        # Store results
        oof_probs[val_idx] = val_probs
        all_test_probs[f"fold_{fold_idx}"] = test_probs
        
        print(f"Fold {fold_idx + 1} complete.")

    print("Step 4: Ensembling predictions...")
    # ensemble expects Dict[name, Probs] and y_val for evaluation
    # We pass the full OOF and the full y_train for scoring
    all_val_preds = {"oof_aggregate": oof_probs}
    final_test_labels = ensemble(all_val_preds, all_test_probs, y_train_full)
    
    print("Step 5: Calculating prediction statistics...")
    oof_labels = np.argmax(oof_probs, axis=1)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_labels)),
            "std": float(np.std(oof_labels)),
            "min": float(np.min(oof_labels)),
            "max": float(np.max(oof_labels)),
        },
        "test": {
            "mean": float(np.mean(final_test_labels)),
            "std": float(np.std(final_test_labels)),
            "min": float(np.min(final_test_labels)),
            "max": float(np.max(final_test_labels)),
        }
    }
    
    print("Step 6: Generating submission file...")
    submission_df = pd.DataFrame({
        "Id": test_ids,
        "Category": final_test_labels
    })
    
    # Ensure all test samples are present, though load_data/preprocess handle this
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    # Save OOF predictions for downstream analysis or meta-learning
    oof_file_path = os.path.join(OUTPUT_DATA_PATH, "oof_probs.npy")
    np.save(oof_file_path, oof_probs)
    
    print(f"Workflow complete. Submission saved to {submission_file_path}")
    
    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "oof_probs_path": oof_file_path
    }
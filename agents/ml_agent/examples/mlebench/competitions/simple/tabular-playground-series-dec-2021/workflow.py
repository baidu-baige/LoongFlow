import os
import gc
import pandas as pd
import numpy as np
from typing import Dict

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/477e9955-ebee-46f4-96b0-878df6f022f5/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting production workflow...")
    
    # 1. Load full dataset
    # validation_mode=False ensures we use the complete 3.6M+ row dataset
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # Uses StratifiedKFold (5-fold) as defined in technical specification
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Containers for out-of-fold and test predictions
    all_val_preds = {}
    all_test_preds = {}
    
    # Define classes for index-to-label mapping (as defined in ensemble.py)
    classes = np.array([1, 2, 3, 4, 6, 7], dtype=np.int32)
    num_classes = len(classes)
    
    # Array to store all OOF probabilities for global stats calculation
    oof_probs = np.zeros((len(X_train_full), num_classes), dtype=np.float32)
    
    # 3. 5-Fold Cross-Validation Loop
    print(f"Starting 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"--- Processing Fold {fold + 1}/5 ---")
        
        # Split data for the current fold
        X_tr, X_va = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_va = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        # a. Apply preprocessing to this fold
        # Engineering spatial features and dropping constant columns
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_tr, y_tr, X_va, y_va, X_test_full
        )
        
        # b. Train model and collect predictions (probabilities)
        # Using the XGBoost GPU engine for efficiency on 3.6M rows
        val_probs, test_probs = PREDICTION_ENGINES["xgb_gpu"](
            X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        )
        
        # c. Store results
        all_val_preds[f"fold_{fold}"] = val_probs
        all_test_preds[f"fold_{fold}"] = test_probs
        oof_probs[val_idx] = val_probs
        
        # d. Memory Management: Explicitly clear large objects and trigger GC
        del X_tr, X_va, y_tr, y_va, X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        gc.collect()
        print(f"Fold {fold + 1} completed and memory cleared.")

    # 4. Ensemble predictions from all folds
    # Uses Soft Voting (probability averaging) as defined in ensemble.py
    # We pass y_train_full as the ground truth reference
    final_test_preds = ensemble(all_val_preds, all_test_preds, y_train_full)
    
    # 5. Compute prediction statistics for the final deliverables
    # Calculate OOF labels from probabilities
    oof_labels = classes[np.argmax(oof_probs, axis=1)]
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_labels)),
            "std": float(np.std(oof_labels)),
            "min": float(np.min(oof_labels)),
            "max": float(np.max(oof_labels)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }
    
    # 6. Generate deliverables
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'Cover_Type': final_test_preds
    })
    
    # Save submission file to specified output path
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to: {submission_file_path}")
    
    # Save meta-artifacts (OOF and Test predictions) for potential downstream analysis
    np.save(os.path.join(OUTPUT_DATA_PATH, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(OUTPUT_DATA_PATH, "test_preds.npy"), final_test_preds)

    # 7. Return production information
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "classes_modeled": classes.tolist(),
        "n_folds": 5
    }
    
    print("Workflow execution complete.")
    return output_info
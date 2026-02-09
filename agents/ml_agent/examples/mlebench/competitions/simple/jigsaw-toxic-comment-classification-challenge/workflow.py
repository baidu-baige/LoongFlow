import os
import numpy as np
import pandas as pd
from typing import Dict

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/4d08636e-bf37-40e0-b9d7-8ffb77d57ea2/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    # 1. Load full dataset with load_data(validation_mode=False)
    print("Step 1: Loading full dataset...")
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy with get_splitter()
    print("Step 2: Initializing data splitter...")
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Storage for cross-validation results
    all_test_preds = {}
    full_oof_preds = np.zeros(y_train_full.shape, dtype=np.float32)
    full_oof_targets = y_train_full.values.astype(np.float32)
    
    # 3. For each fold:
    num_splits = splitter.get_n_splits()
    print(f"Step 3: Starting cross-validation training on {num_splits} folds...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n--- Processing Fold {fold_idx + 1}/{num_splits} ---")
        
        # a. Split train/validation data
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        # b. Apply preprocess() to this fold
        print(f"Tokenizing data for Fold {fold_idx + 1}...")
        X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p = preprocess(X_tr, y_tr, X_val, y_val, X_test_full)
        
        # c. Train each model and collect val + test predictions
        # Using DeBERTa-v3 as the primary engine
        print(f"Training DeBERTa-v3 for Fold {fold_idx + 1}...")
        val_preds, test_preds = PREDICTION_ENGINES["deberta_v3"](X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p)
        
        # Store fold results
        fold_name = f"fold_{fold_idx}"
        all_test_preds[fold_name] = test_preds
        full_oof_preds[val_idx] = val_preds
        
    # 4. Ensemble predictions from all models
    # We pass the full OOF matrix as a single entry to the ensemble for scoring purposes,
    # and all fold-wise test predictions for averaging.
    print("\nStep 4: Performing ensemble averaging...")
    all_val_preds_wrapper = {"OOF_Combined": full_oof_preds}
    final_test_preds = ensemble(all_val_preds_wrapper, all_test_preds, full_oof_targets)
    
    # 5. Compute prediction statistics
    print("Step 5: Calculating prediction statistics...")
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(full_oof_preds)),
            "std": float(np.std(full_oof_preds)),
            "min": float(np.min(full_oof_preds)),
            "max": float(np.max(full_oof_preds)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }
    
    # 6. Generate deliverables (submission file)
    print("Step 6: Generating final submission file...")
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    submission_df = pd.DataFrame(final_test_preds, columns=target_columns)
    submission_df.insert(0, 'id', test_ids.values)
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    
    # 7. Save artifacts and return metadata
    print(f"Workflow completed successfully. Submission saved to: {submission_file_path}")
    
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
    }
    
    return output_info
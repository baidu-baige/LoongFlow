import os
import numpy as np
import pandas as pd
import gc
import torch
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/02d42284-9bf3-4f97-ab6c-7ea839095b54/3/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.

    This function integrates all pipeline components (data loading, preprocessing, 
    data splitting, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    """
    print("Execution: workflow (Stage 6)")
    
    # Ensure the output directory exists for artifacts
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    # Returns features (X), targets (y) for training, and features, IDs for testing
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    # Uses GroupKFoldShuffled based on the 'anchor' column to prevent leakage
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Initialize containers for predictions and Out-of-Fold (OOF) tracking
    all_val_preds = {}
    all_test_preds = {}
    oof_predictions = np.zeros(len(y_train_full))
    
    # Identify the training engine from the registry
    model_key = "deberta_v3_large_v2"
    if model_key not in PREDICTION_ENGINES:
        raise KeyError(f"Model engine '{model_key}' not found in registry.")
    train_fn = PREDICTION_ENGINES[model_key]

    # 3. 5-Fold Cross-Validation Loop
    print(f"Starting 5-fold cross-validation using {model_key}...")
    
    # The splitter generates indices for train and validation sets for each fold
    fold_generator = splitter.split(X_train_full, y_train_full)
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_generator):
        print(f"\n--- Processing Fold {fold_idx} ---")
        
        # Split raw dataframe based on current fold indices
        X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        # a. Preprocess data for this fold
        # Converts text pairs into tokenized sequences (max_length=172)
        X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p = preprocess(
            X_tr, y_tr, X_val, y_val, X_test_full
        )
        
        # b. Train and Predict
        # Executes training on multi-GPU DDP and returns sigmoid-activated predictions [0, 1]
        val_preds, test_preds = train_fn(
            X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p
        )
        
        # c. Store results for ensembling and OOF analysis
        all_val_preds[f"fold_{fold_idx}"] = val_preds
        all_test_preds[f"fold_{fold_idx}"] = test_preds
        oof_predictions[val_idx] = val_preds
        
        # Cleanup to manage memory and GPU resources effectively
        del X_tr, y_tr, X_val, y_val, X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Ensemble predictions from all folds
    # Calculates arithmetic mean of test predictions across folds and clips to valid range [0, 1]
    final_test_preds = ensemble(
        all_val_preds=all_val_preds,
        all_test_preds=all_test_preds,
        y_val=y_train_full
    )

    # 5. Compute prediction distribution statistics
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_predictions)),
            "std": float(np.std(oof_predictions)),
            "min": float(np.min(oof_predictions)),
            "max": float(np.max(oof_predictions)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }

    # 6. Generate submission file
    submission_df = pd.DataFrame({
        'id': test_ids,
        'score': final_test_preds
    })
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to: {submission_file_path}")

    # 7. Construct and return the deliverables dictionary
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
        "n_folds": 5,
        "model_used": model_key
    }
    
    print("\nWorkflow completed successfully.")
    return output_info

if __name__ == "__main__":
    # Internal execution check
    results = workflow()
    print(results)
import os
import pandas as pd
import numpy as np
import torch
import gc
from typing import Dict

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/e81e4df9-fbfb-4465-8b24-4be8ee1f51f4/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting production workflow...")
    
    # 1. Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 2. Load full dataset
    # validation_mode=False ensures we use the complete dataset for production
    X_train_full, y_train_full, X_test, test_ids = load_data(validation_mode=False)
    
    # 3. Set up data splitting strategy
    splitter = get_splitter(X_train_full, y_train_full)
    num_splits = splitter.get_n_splits()
    
    # Initialize containers for out-of-fold (OOF) and test predictions
    # We store probabilities for each model to be used in the ensemble stage
    model_names = list(PREDICTION_ENGINES.keys())
    oof_probs_all_models = {name: np.zeros((len(X_train_full), 6), dtype=np.float32) for name in model_names}
    test_probs_all_models = {name: np.zeros((len(X_test), 6), dtype=np.float32) for name in model_names}

    # 4. Cross-Validation Loop
    print(f"Starting {num_splits}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"--- Processing Fold {fold + 1}/{num_splits} ---")
        
        # Split raw data for this fold
        X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full[train_idx]
        X_vl, y_vl = X_train_full.iloc[val_idx], y_train_full[val_idx]
        
        # 5. Preprocess data for this fold
        X_tr_p, y_tr_p, X_vl_p, y_vl_p, X_te_p = preprocess(X_tr, y_tr, X_vl, y_vl, X_test)
        
        # 6. Train each registered model engine
        for name, train_fn in PREDICTION_ENGINES.items():
            print(f"Training model: {name}")
            val_preds, test_preds = train_fn(X_tr_p, y_tr_p, X_vl_p, y_vl_p, X_te_p)
            
            # Store OOF predictions
            oof_probs_all_models[name][val_idx] = val_preds
            # Incrementally average test predictions across folds
            test_probs_all_models[name] += test_preds / num_splits
            
        # Clear fold-specific data to optimize memory
        del X_tr_p, y_tr_p, X_vl_p, y_vl_p, X_te_p
        gc.collect()
        torch.cuda.empty_cache()

    # 7. Ensemble Predictions
    # The ensemble function will optimize class-wise thresholds on the full OOF matrix
    print("Entering ensemble stage...")
    final_test_labels = ensemble(
        all_val_preds=oof_probs_all_models,
        all_test_preds=test_probs_all_models,
        y_val=y_train_full
    )

    # 8. Generate Submission File
    submission_df = pd.DataFrame({
        "image": test_ids,
        "labels": final_test_labels
    })
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")

    # 9. Compute Prediction Statistics
    # We calculate stats based on the mean probabilities across all models
    combined_oof_probs = np.mean(list(oof_probs_all_models.values()), axis=0)
    combined_test_probs = np.mean(list(test_probs_all_models.values()), axis=0)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(combined_oof_probs)),
            "std": float(np.std(combined_oof_probs)),
            "min": float(np.min(combined_oof_probs)),
            "max": float(np.max(combined_oof_probs)),
        },
        "test": {
            "mean": float(np.mean(combined_test_probs)),
            "std": float(np.std(combined_test_probs)),
            "min": float(np.min(combined_test_probs)),
            "max": float(np.max(combined_test_probs)),
        }
    }

    # 10. Prepare final deliverables
    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "num_folds": num_splits,
        "models_trained": model_names
    }

    print("Workflow execution completed successfully.")
    return output_info
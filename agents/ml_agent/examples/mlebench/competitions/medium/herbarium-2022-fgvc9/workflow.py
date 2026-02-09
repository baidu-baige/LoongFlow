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

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/herbarium-2022-fgvc9/prepared/public"
OUTPUT_DATA_PATH = "output/5bfa936c-0be4-4e88-95de-92261403881f/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    print("Starting production workflow...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load full dataset
    print("Stage 1: Loading full dataset...")
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    print("Stage 2: Setting up data splitter...")
    splitter = get_splitter(X_train_full, y_train_full)

    # 3. Execute training for Fold 0 (as specified in technical baseline)
    print("Stage 3: Beginning single-fold training (Fold 0)...")
    all_val_preds = {}
    all_test_preds = {}
    
    # HerbariumSplitter is designed to yield only Fold 0
    for train_idx, val_idx in splitter.split(X_train_full, y_train_full):
        X_tr, X_va = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_va = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        # Apply preprocessing
        print("Preprocessing current fold...")
        X_tr_proc, y_tr_proc, X_va_proc, y_va_proc, X_te_proc = preprocess(
            X_tr, y_tr, X_va, y_va, X_test_full
        )
        
        # Train and Predict using the ConvNeXt-MTL engine
        model_name = "convnext_mtl"
        print(f"Training model engine: {model_name}...")
        train_fn = PREDICTION_ENGINES[model_name]
        val_preds, test_preds = train_fn(
            X_tr_proc, y_tr_proc, X_va_proc, y_va_proc, X_te_proc
        )
        
        all_val_preds[model_name] = val_preds
        all_test_preds[model_name] = test_preds
        
        # Capture y_val_processed for ensemble evaluation
        y_val_final = y_va_proc
        break # Ensure only Fold 0 is processed as per technical spec requirements

    # 4. Ensemble predictions
    print("Stage 4: Ensembling predictions...")
    final_test_preds = ensemble(all_val_preds, all_test_preds, y_val_final)

    # 5. Compute prediction statistics
    print("Stage 5: Computing prediction statistics...")
    # Using the first model's OOF predictions for stats
    oof_preds = list(all_val_preds.values())[0]
    
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

    # 6. Generate deliverables
    print("Stage 6: Generating final deliverables...")
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'Predicted': final_test_preds
    })
    
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")

    # Step 7: Final check and return paths
    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
    }
    
    print("Workflow execution complete.")
    return output_info
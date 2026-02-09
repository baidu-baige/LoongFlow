import pandas as pd
import numpy as np
import os
import torch
import gc
from typing import Dict, Any

# Import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Define paths as per technical specification
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/lmsys-chatbot-arena/prepared/public"
OUTPUT_DATA_PATH = "output/e051fb32-5ec2-4424-8b42-87dd7b28dacc/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    # 1. Load full dataset
    print("Stage 1: Loading full dataset...")
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)

    # 2. Set up data splitting strategy
    print("Stage 2: Initializing data splitter (Stratified 5-Fold)...")
    splitter = get_splitter(X_train_full, y_train_full)
    
    # Initialize containers for out-of-fold and test predictions
    all_val_preds = {}
    all_test_preds = {}
    oof_preds = np.zeros((len(X_train_full), 3))
    
    train_engine = PREDICTION_ENGINES["deberta_v3_large"]
    
    # 3. Cross-Validation Loop
    print(f"Stage 3: Starting 5-fold cross-validation training and inference...")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n--- Processing Fold {fold} ---")
        
        # Split data for this fold
        X_train_f = X_train_full.iloc[train_idx]
        y_train_f = y_train_full.iloc[train_idx]
        X_val_f = X_train_full.iloc[val_idx]
        y_val_f = y_train_full.iloc[val_idx]
        
        # a. Preprocess this fold
        # Preprocess logic handles tokenization and truncation specific to the task
        X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p = preprocess(
            X_train_f, y_train_f, X_val_f, y_val_f, X_test_full
        )
        
        # b. Train model and predict
        # This function utilizes 2-GPU DDP and returns probabilities
        val_preds_f, test_preds_f = train_engine(
            X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p
        )
        
        # Store results
        # val_preds_f is shape (len(val_idx), 3)
        oof_preds[val_idx] = val_preds_f
        all_test_preds[f"fold_{fold}"] = test_preds_f
        
        # Calculate individual fold log loss for logging
        from sklearn.metrics import log_loss
        fold_loss = log_loss(y_val_f, val_preds_f)
        print(f"Fold {fold} Validation Log Loss: {fold_loss:.6f}")
        
        # Memory Management: Clear variables and GPU cache after each fold
        del X_tr_p, y_tr_p, X_va_p, y_va_p, X_te_p, val_preds_f, test_preds_f
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Ensemble stage
    print("\nStage 4: Ensembling predictions...")
    # We pass the full OOF array as a single entry to evaluate overall CV log loss
    combined_val_preds = {"full_oof": oof_preds}
    final_test_preds = ensemble(
        all_val_preds=combined_val_preds,
        all_test_preds=all_test_preds,
        y_val=y_train_full.values
    )

    # 5. Generate submission file
    print("Stage 5: Generating final submission artifact...")
    submission_df = pd.DataFrame({
        "id": test_ids,
        "winner_model_a": final_test_preds[:, 0],
        "winner_model_b": final_test_preds[:, 1],
        "winner_tie": final_test_preds[:, 2]
    })
    
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")

    # 6. Compute prediction statistics
    # Primitive types only for JSON serialization
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

    # 7. Final Deliverables
    output_info = {
        "submission_file_path": submission_path,
        "prediction_stats": prediction_stats,
        "oof_log_loss": float(log_loss(y_train_full, oof_preds)),
    }
    
    print("\nFull pipeline execution completed successfully.")
    return output_info

if __name__ == "__main__":
    results = workflow()
    print(results)
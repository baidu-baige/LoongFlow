import os
import numpy as np
import pandas as pd
from typing import Dict
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

# Path constants for the environment
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/spooky-author-identification/prepared/public"
OUTPUT_DATA_PATH = "output/368bc6e8-482c-48b8-a870-040b0c3a264c/6/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This function integrates data loading, 5-fold stratified cross-validation, 
    multi-modal preprocessing (Transformers, TF-IDF, spaCy), calibrated model training, 
    and optimized ensemble weighting to generate the final competition submission.
    """
    print("Initiating production pipeline for Spooky Author Identification.")

    # 1. Load full dataset
    # validation_mode=False is mandatory for production artifacts.
    X_train_full, y_train_full, X_test_full, test_ids = load_data(validation_mode=False)
    
    # 2. Set up data splitting strategy
    # StratifiedKFold maintains class proportions for EAP, HPL, and MWS across folds.
    splitter = get_splitter(X_train_full, y_train_full)
    n_splits = splitter.get_n_splits()
    
    # Initialize containers for Out-of-Fold (OOF) and Test predictions
    n_train = len(X_train_full)
    n_test = len(X_test_full)
    model_names = sorted(list(PREDICTION_ENGINES.keys()))
    
    # Dicts to store OOF and Test probabilities for each model
    all_val_preds = {name: np.zeros((n_train, 3)) for name in model_names}
    all_test_preds_accum = {name: np.zeros((n_test, 3)) for name in model_names}
    y_oof_true = np.zeros(n_train, dtype=int)
    
    # 3. Cross-Validation Loop
    # Execute 5-fold CV to obtain robust OOF predictions for ensemble weight optimization.
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        print(f"\n--- Executing CV Fold {fold_idx + 1} / {n_splits} ---")
        
        # Segment data for this fold
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        # a. Preprocess fold data
        # Generates Transformer IDs/masks, Pruned TF-IDF, and Dense Meta-features.
        X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p = preprocess(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_full
        )
        
        # Capture integer labels for OOF evaluation (0: EAP, 1: HPL, 2: MWS)
        y_oof_true[val_idx] = y_val_p
        
        # b. Train and Predict with each engine in the registry
        for name in model_names:
            print(f"  Training Engine: {name}")
            # Each engine handles its own training logic (DDP for Transformers, etc.)
            val_p, te_p = PREDICTION_ENGINES[name](
                X_tr_p, y_tr_p, X_val_p, y_val_p, X_te_p
            )
            
            # Store OOF probabilities at original indices
            all_val_preds[name][val_idx] = val_p
            # Accumulate test probabilities for averaging over folds
            all_test_preds_accum[name] += te_p / float(n_splits)

    # 4. Ensemble Stage
    # Optimizes weights for the 5-pillar ensemble using SLSQP on OOF predictions.
    print("\nExecuting ensemble weight optimization and final test generation...")
    final_test_probs = ensemble(all_val_preds, all_test_preds_accum, y_oof_true)
    
    # 5. Generate Submission File
    # Ensure columns match competition requirements: id, EAP, HPL, MWS.
    # LabelEncoder in preprocess ensures alphabetic class index mapping.
    submission_df = pd.DataFrame(final_test_probs, columns=['EAP', 'HPL', 'MWS'])
    submission_df.insert(0, 'id', test_ids.values)
    
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Production submission saved to: {submission_file_path}")

    # 6. Compute Prediction Statistics
    # Calculate OOF distribution across all models as a performance proxy.
    oof_stack = np.concatenate([all_val_preds[name] for name in model_names], axis=0)
    
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_stack)),
            "std": float(np.std(oof_stack)),
            "min": float(np.min(oof_stack)),
            "max": float(np.max(oof_stack)),
        },
        "test": {
            "mean": float(np.mean(final_test_probs)),
            "std": float(np.std(final_test_probs)),
            "min": float(np.min(final_test_probs)),
            "max": float(np.max(final_test_probs)),
        }
    }

    print("Pipeline execution complete. Deliverables generated successfully.")
    
    return {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
    }

if __name__ == "__main__":
    workflow()
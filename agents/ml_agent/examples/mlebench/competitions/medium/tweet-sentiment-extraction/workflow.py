import os
import pandas as pd
import numpy as np
from typing import Dict

# import all component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/tweet-sentiment-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/1cc1b6ac-193c-4f3f-a388-380b832f53e8/5/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.

    This function integrates all pipeline components (data loading, preprocessing, 
    data splitting, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    """
    # Step 0: Environment Setup
    print("Step 0: Initializing environment and output directories...")
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Step 1: Load full dataset
    # Technical Specification: MUST call load_data(validation_mode=False)
    print("Step 1: Loading full dataset...")
    X_train, y_train, X_test, test_ids = load_data(validation_mode=False)

    # Step 2: Initialize splitter (Stratified 5-Fold)
    # Technical Specification: Strategy uses 5-fold cross-validation
    print("Step 2: Initializing stratified splitter...")
    splitter = get_splitter(X_train, y_train)

    # Step 3: Multi-fold training loop
    # We collect predictions from each fold for ensembling.
    all_val_preds = {}
    all_test_preds = {}
    last_y_val_p = None

    num_folds = splitter.get_n_splits(X_train)
    print(f"Step 3: Starting {num_folds}-fold training loop using DeBERTa-v3-Large...")

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train)):
        print(f"\n--- Processing Fold {fold_idx + 1}/{num_folds} ---")
        
        # a. Data Splitting
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # b. Preprocessing
        # Maps text to DeBERTa tokens and character spans to token indices.
        print(f"Preprocessing fold {fold_idx + 1}...")
        X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p = preprocess(
            X_tr, y_tr, X_val, y_val, X_test
        )
        
        # c. Training and Prediction
        # Uses 'deberta_v3_large_qa' engine which implements DDP across 2 GPUs.
        model_name = "deberta_v3_large_qa"
        train_fn = PREDICTION_ENGINES[model_name]
        
        print(f"Training and extracting logits for fold {fold_idx + 1}...")
        val_preds, test_preds = train_fn(
            X_tr_p, y_tr_p, X_val_p, y_val_p, X_test_p
        )
        
        # Store predictions for the ensemble stage
        fold_key = f"fold_{fold_idx}"
        all_val_preds[fold_key] = val_preds
        all_test_preds[fold_key] = test_preds
        
        # Store y_val_p for the ensemble function's signature
        last_y_val_p = y_val_p
        
        # Memory Management: Clear fold-specific large objects
        del X_tr_p, y_tr_p, X_val_p, y_val_p
        print(f"Fold {fold_idx + 1} processing complete.")

    # Step 4: Ensemble and Decode
    # Combines logits from all folds and applies the Neutral/Short-Text heuristics.
    print("\nStep 4: Ensembling logits and decoding final spans with heuristics...")
    final_test_strings = ensemble(all_val_preds, all_test_preds, last_y_val_p)

    # Step 5: Compute prediction statistics
    # Uses string length as a numerical proxy for distribution analysis.
    print("Step 5: Computing prediction statistics...")
    test_lengths = [len(str(s)) for s in final_test_strings]
    # OOF stats: Using the ground truth selected_text lengths as the reference baseline
    oof_lengths = [len(str(s)) for s in y_train]

    prediction_stats = {
        "oof": {
            "mean": float(np.mean(oof_lengths)),
            "std": float(np.std(oof_lengths)),
            "min": float(np.min(oof_lengths)),
            "max": float(np.max(oof_lengths)),
        },
        "test": {
            "mean": float(np.mean(test_lengths)),
            "std": float(np.std(test_lengths)),
            "min": float(np.min(test_lengths)),
            "max": float(np.max(test_lengths)),
        }
    }

    # Step 6: Generate Submission File
    print("Step 6: Generating final submission.csv...")
    submission_df = pd.DataFrame({
        'textID': test_ids.values,
        'selected_text': final_test_strings
    })
    
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # Step 7: Finalize Artifacts
    print(f"Workflow complete. Submission saved to: {submission_file_path}")
    
    output_info = {
        "submission_file_path": submission_file_path,
        "prediction_stats": prediction_stats,
    }
    
    return output_info

if __name__ == "__main__":
    workflow()
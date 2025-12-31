# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
# Import component functions from their respective modules
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

# Define constants based on task description and requirements
BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/b4d4dfce-4367-41a9-8cac-a59279a6d65f/11/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This function integrates all pipeline components (data loading, feature engineering, 
    cross-validation, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    
    Returns:
        dict: A dictionary containing all task deliverables, specifically the 
              submission file path and model performance metrics.
    """
    print("Initializing Workflow...")

    # --- 0. Setup Environment ---
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # --- 1. Load Data ---
    print("Step 1: Loading Data...")
    # validation_mode=False ensures the complete dataset is loaded for production
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # Validation check on loaded data
    if X.empty or y.empty:
        raise ValueError("Loaded data is empty. Check data paths and load_data logic.")

    # Log data statistics
    n_samples = len(X)
    n_classes = y.shape[1] if hasattr(y, 'shape') else 0
    print(f"   Data Loaded. Train: {n_samples}, Test: {len(X_test)}, Classes: {n_classes}")

    # --- 2. Setup Cross-Validation ---
    print("Step 2: Setting up Cross-Validation Strategy...")
    # Initialize the cross-validation splitter
    cv_splitter = cross_validation(X, y)

    # --- 3. Initialize Prediction Storage ---
    model_names = list(PREDICTION_ENGINES.keys())

    # Dictionary to store Out-Of-Fold (OOF) predictions for validation
    # Keys: Model names, Values: Numpy arrays of shape (n_train_samples, n_classes)
    # Using float64 for precision accumulation
    all_oof_preds = {name: np.zeros((n_samples, n_classes), dtype=np.float64) for name in model_names}

    # Dictionary to store Test predictions
    # Keys: Model names, Values: List of numpy arrays (one per fold)
    all_test_preds = {name: [] for name in model_names}

    # --- 4. Main CV Loop ---
    print("Step 3: Starting Cross-Validation Loop...")

    # Iterate through each fold defined by the splitter
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        print(f"   Processing Fold {fold_idx + 1}...")

        # a. Split Train/Validation Data
        # Using iloc ensures correct slicing by integer position provided by the splitter
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # b. Feature Engineering
        # Creates features, merges MIML segment data, and applies scaling based on training fold
        # X_test is passed here to be transformed based on statistics of X_train_fold
        X_train_trans, y_train_trans, X_val_trans, y_val_trans, X_test_trans = create_features(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            X_test
        )

        # c. Model Training & Prediction
        for model_name, engine_fn in PREDICTION_ENGINES.items():
            # print(f"      Running Model: {model_name}")

            # Train the model and get predictions for validation and test sets
            val_preds, test_preds = engine_fn(
                X_train_trans, y_train_trans,
                X_val_trans, y_val_trans,
                X_test_trans
            )

            # Ensure predictions are numpy arrays before storage
            if hasattr(val_preds, 'values'):
                val_preds = val_preds.values
            if hasattr(test_preds, 'values'):
                test_preds = test_preds.values

            # Store OOF predictions (fill indices corresponding to validation set)
            all_oof_preds[model_name][val_idx] = val_preds

            # Collect Test predictions for this fold
            all_test_preds[model_name].append(test_preds)

    # --- 5. Ensemble ---
    print("Step 4: Ensembling Predictions...")
    # Aggregates test predictions across folds (mean) and across models (mean)
    final_test_predictions = ensemble(all_oof_preds, all_test_preds, y)

    # --- 6. Scoring (Internal Validation) ---
    print("Step 5: Calculating Metrics...")
    model_scores = {}

    # Convert ground truth to numpy for scoring
    y_true_np = y.values if hasattr(y, 'values') else np.array(y)

    # Calculate OOF AUC for each individual model
    for m_name, preds in all_oof_preds.items():
        try:
            # Replace NaNs with 0 in case of dropped folds (safety)
            preds_clean = np.nan_to_num(preds)
            # Use macro average for multi-label classification AUC
            score = roc_auc_score(y_true_np, preds_clean, average='macro')
            model_scores[f"{m_name}_oof_auc"] = float(score)
        except Exception as e:
            print(f"   Warning: Scoring failed for {m_name}: {e}")
            model_scores[f"{m_name}_oof_auc"] = 0.0

    # Calculate Ensemble OOF AUC (Simulated)
    try:
        # Average the OOF predictions from all models to approximate ensemble performance
        oof_stack = np.array([np.nan_to_num(all_oof_preds[m]) for m in model_names])
        if oof_stack.size > 0:
            mean_oof = np.mean(oof_stack, axis=0)
            ensemble_score = roc_auc_score(y_true_np, mean_oof, average='macro')
            model_scores["ensemble_oof_auc"] = float(ensemble_score)
            print(f"   Ensemble OOF Macro AUC: {ensemble_score:.5f}")
    except Exception as e:
        print(f"   Warning: Ensemble scoring failed: {e}")
        model_scores["ensemble_oof_auc"] = 0.0

    # --- 7. Format Submission ---
    print("Step 6: Formatting Submission...")

    submission_rows = []
    # Ensure test_ids is a flat numpy array
    test_ids_arr = test_ids.values.flatten() if hasattr(test_ids, 'values') else np.array(test_ids).flatten()

    # Validation
    if final_test_predictions.shape[0] != len(test_ids_arr):
        print(
            f"   Warning: Prediction shape {final_test_predictions.shape} differs from IDs length {len(test_ids_arr)}")

    num_species = final_test_predictions.shape[1]
    n_test_samples = len(test_ids_arr)

    # Construct submission rows: Id = rec_id * 100 + species_idx
    # This loop logic matches the task description exactly
    for i in range(n_test_samples):
        if i >= len(final_test_predictions):
            break

        rec_id = test_ids_arr[i]
        probs = final_test_predictions[i]

        for species_idx in range(num_species):
            prob = probs[species_idx]
            # Specific ID requirement from task description
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({
                "Id": row_id,
                "Probability": float(prob)
            })

    submission_df = pd.DataFrame(submission_rows)

    # --- 8. Save Artifacts ---
    print("Step 7: Saving Artifacts...")

    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"   Submission saved to: {submission_path}")

    # --- 9. Return Results ---
    output_info = {
        "submission_file_path": submission_path,
        "model_scores": model_scores,
        "metrics": {
            "n_train": n_samples,
            "n_test": len(X_test),
            "n_folds": cv_splitter.get_n_splits(X, y)
        }
    }

    print("Workflow Completed Successfully.")
    return output_info

# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
# Assume all component functions are available for import
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/2ea9a3e6-0185-40d8-bd93-7afe244a50a1/2/executor/output"


def workflow() -> dict:
    """
    Executes the complete machine learning workflow to generate the required deliverables.
    This function orchestrates the entire pipeline, from data processing to model
    training and evaluation. Its primary purpose is to produce and consolidate all
    the final artifacts specified in the task description.

    Returns:
        dict: A dictionary containing all the deliverables required by the task 
              description. It serves as a structured, serializable manifest of the results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Step 1: Load data
    print("Step 1: Loading data...")
    X, y, X_test, test_ids = load_data()
    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target shape: {y.shape}")

    # Step 2: Set up cross-validation strategy
    print("\nStep 2: Setting up cross-validation...")
    cv = cross_validation(X, y)

    # Step 3: Initialize storage for predictions
    print("\nStep 3: Initializing prediction storage...")
    n_train = X.shape[0]
    n_test = X_test.shape[0]
    n_targets = y.shape[1]
    target_columns = y.columns.tolist()

    # Dictionary to store OOF predictions for each model
    all_oof_preds = {}
    # Dictionary to store test predictions for each model (list of fold predictions)
    all_test_preds = {}

    # Initialize storage for each model
    for model_name in PREDICTION_ENGINES.keys():
        all_oof_preds[model_name] = np.zeros((n_train, n_targets))
        all_test_preds[model_name] = []

    # Store transformed y for ensemble function
    y_transformed_full = np.zeros((n_train, n_targets))

    # Step 4: Cross-validation loop
    print("\nStep 4: Running cross-validation...")
    fold_scores = {model_name: [] for model_name in PREDICTION_ENGINES.keys()}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1} ---")

        # Step 4a: Split train/validation data
        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        y_train_fold = y.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)
        y_val_fold = y.iloc[val_idx].reset_index(drop=True)
        X_test_fold = X_test.reset_index(drop=True)

        print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")

        # Step 4b: Create features for training data
        # The create_features function transforms X_train, y_train, and X_test
        # We need to handle validation data separately
        print("Creating features...")

        # First, create features using training data as reference
        X_train_transformed, y_train_transformed, X_test_transformed = create_features(
            X_train_fold, y_train_fold, X_test_fold
        )

        # For validation, we need to transform it using the same approach
        # We'll concatenate val with test temporarily to ensure consistent feature engineering
        # Then extract the validation portion

        # Create a combined dataset for validation transformation
        # Use X_train_fold as reference for feature engineering on X_val_fold
        X_val_transformed, y_val_transformed, _ = create_features(
            X_train_fold, y_val_fold, X_val_fold
        )

        # The y_val_transformed is now log1p transformed
        # X_val_transformed is the validation features (returned as the "test" output)
        # We need to use the third return value which is the transformed X_val_fold

        # Actually, looking at create_features more carefully:
        # It returns (X_train_transformed, y_train_transformed, X_test_transformed)
        # So for validation, we pass X_val_fold as X_test and get it back transformed

        # Re-do this correctly:
        # For validation features, we need to pass X_val_fold as the "test" set
        _, y_val_transformed_temp, X_val_transformed = create_features(
            X_train_fold, y_val_fold, X_val_fold
        )

        # y_val_transformed_temp is the log1p of y_val_fold
        y_val_transformed = y_val_transformed_temp

        # Store transformed y values for ensemble (in log space)
        y_transformed_full[val_idx] = y_val_transformed.values

        # Step 4c: Train each model
        for model_name, train_fn in PREDICTION_ENGINES.items():
            print(f"Training {model_name}...")

            # Train and predict
            val_preds, test_preds = train_fn(
                X_train_transformed,
                y_train_transformed,
                X_val_transformed,
                y_val_transformed,
                X_test_transformed
            )

            # Store OOF predictions (in log space)
            if isinstance(val_preds, pd.DataFrame):
                all_oof_preds[model_name][val_idx] = val_preds.values
            else:
                all_oof_preds[model_name][val_idx] = np.array(val_preds)

            # Append test predictions to list
            all_test_preds[model_name].append(test_preds)

            # Calculate fold RMSLE (in log space, this is RMSE)
            val_preds_np = val_preds.values if isinstance(val_preds, pd.DataFrame) else np.array(val_preds)
            y_val_np = y_val_transformed.values if isinstance(y_val_transformed, pd.DataFrame) else np.array(
                y_val_transformed)

            fold_rmse = np.sqrt(np.mean((val_preds_np - y_val_np) ** 2))
            fold_scores[model_name].append(fold_rmse)
            print(f"  {model_name} Fold {fold_idx + 1} RMSE (log space): {fold_rmse:.6f}")

    # Step 5: Report mean CV scores
    print("\n" + "=" * 50)
    print("Step 5: Cross-validation Results")
    print("=" * 50)

    model_cv_scores = {}
    for model_name, scores in fold_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        model_cv_scores[model_name] = float(mean_score)
        print(f"{model_name}: Mean RMSE (log space) = {mean_score:.6f} (+/- {std_score:.6f})")

    # Step 6 & 7: Ensemble predictions
    print("\nStep 6 & 7: Ensembling predictions...")

    # Convert OOF predictions to DataFrames
    all_oof_preds_df = {}
    for model_name, oof_preds in all_oof_preds.items():
        all_oof_preds_df[model_name] = pd.DataFrame(oof_preds, columns=target_columns)

    # Convert test predictions lists to proper format
    all_test_preds_df = {}
    for model_name, test_preds_list in all_test_preds.items():
        all_test_preds_df[model_name] = [
            pred if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred, columns=target_columns)
            for pred in test_preds_list
        ]

    # Create y_true_full DataFrame (in log space)
    y_true_full = pd.DataFrame(y_transformed_full, columns=target_columns)

    # Call ensemble function
    final_predictions = ensemble(all_oof_preds_df, all_test_preds_df, y_true_full)

    # Step 8: Create submission DataFrame
    print("\nStep 8: Creating submission file...")

    # Ensure predictions are non-negative
    final_predictions = final_predictions.clip(lower=0)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_ids.values,
        'formation_energy_ev_natom': final_predictions['formation_energy_ev_natom'].values,
        'bandgap_energy_ev': final_predictions['bandgap_energy_ev'].values
    })

    # Ensure id is integer
    submission['id'] = submission['id'].astype(int)

    # Step 9: Save submission file
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print("\nSubmission preview:")
    print(submission.head())
    print("\nSubmission statistics:")
    print(submission.describe())

    # Calculate overall CV score (mean of both targets)
    overall_cv_score = np.mean(list(model_cv_scores.values()))

    # Create output info dictionary
    output_info = {
        "submission_file_path": submission_path,
        "model_scores": model_cv_scores,
        "overall_cv_score": float(overall_cv_score),
        "n_train_samples": int(n_train),
        "n_test_samples": int(n_test),
        "n_features": int(X.shape[1]),
        "n_folds": 5,
        "models_used": list(PREDICTION_ENGINES.keys()),
        "target_columns": target_columns
    }

    print("\n" + "=" * 50)
    print("Workflow completed successfully!")
    print("=" * 50)
    print(f"Overall CV Score (RMSE in log space): {overall_cv_score:.6f}")

    return output_info

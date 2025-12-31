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
# Assume all component functions are available for import
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/aerial-cactus-identification/prepared/public"
OUTPUT_DATA_PATH = "output/153e4624-940b-4d19-a37d-90435531bfd1/1/executor/output"


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
    # Step 1: Load data using load_data module
    print("Step 1: Loading data...")
    X, y, X_test, test_ids = load_data()
    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of test IDs: {len(test_ids)}")

    # Step 2: Create cross-validation splits using cross_validation module
    print("\nStep 2: Setting up cross-validation...")
    cv = cross_validation(X, y)

    # Step 3: Initialize storage for predictions
    # OOF predictions: dict of {model_name: array of predictions}
    # Test predictions: dict of {model_name: list of fold predictions}
    all_oof_preds = {}
    all_test_preds = {}

    # Get model names from PREDICTION_ENGINES
    model_names = list(PREDICTION_ENGINES.keys())
    print(f"Models to train: {model_names}")

    # Initialize storage for each model
    for model_name in model_names:
        all_oof_preds[model_name] = np.zeros(len(X))
        all_test_preds[model_name] = []

    # Track fold scores for monitoring
    fold_scores = {model_name: [] for model_name in model_names}

    # Step 4: OUTER LOOP - for each fold
    print("\nStep 3: Training models with cross-validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold_idx + 1}")
        print(f"{'=' * 50}")

        # Split train/validation data
        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        y_train_fold = y.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)
        y_val_fold = y.iloc[val_idx].reset_index(drop=True)

        print(f"Train size: {len(X_train_fold)}, Validation size: {len(X_val_fold)}")

        # Apply feature engineering using create_features module
        print("Applying feature engineering...")
        X_train_transformed, y_train_transformed, X_val_transformed = create_features(
            X_train_fold, y_train_fold, X_val_fold
        )

        # Also transform test data for this fold
        _, _, X_test_transformed = create_features(
            X_train_fold, y_train_fold, X_test
        )

        # INNER LOOP: for each model
        for model_name in model_names:
            print(f"\nTraining model: {model_name}")

            # Get the training function
            train_fn = PREDICTION_ENGINES[model_name]

            # Train and predict
            val_preds, test_preds = train_fn(
                X_train_transformed,
                y_train_transformed,
                X_val_transformed,
                y_val_fold,
                X_test_transformed
            )

            # Store OOF predictions
            all_oof_preds[model_name][val_idx] = val_preds

            # Append test predictions to list
            all_test_preds[model_name].append(test_preds)

            # Calculate and store fold AUC-ROC score
            fold_auc = roc_auc_score(y_val_fold, val_preds)
            fold_scores[model_name].append(fold_auc)
            print(f"Fold {fold_idx + 1} - {model_name} AUC-ROC: {fold_auc:.6f}")

    # Print overall CV scores
    print(f"\n{'=' * 50}")
    print("Cross-Validation Results Summary")
    print(f"{'=' * 50}")
    model_cv_scores = {}
    for model_name in model_names:
        mean_score = np.mean(fold_scores[model_name])
        std_score = np.std(fold_scores[model_name])
        model_cv_scores[model_name] = mean_score
        print(f"{model_name}: Mean AUC-ROC = {mean_score:.6f} (+/- {std_score:.6f})")

    # Step 5: Ensemble predictions
    print("\nStep 4: Ensembling predictions...")
    final_test_predictions = ensemble(all_oof_preds, all_test_preds, y)

    # Step 6: Create submission DataFrame
    print("\nStep 5: Creating submission file...")
    submission_df = pd.DataFrame({
        'id': test_ids,
        'has_cactus': final_test_predictions
    })

    # Ensure proper formatting
    submission_df['has_cactus'] = submission_df['has_cactus'].astype(float)

    # Sort by id to match sample submission format
    submission_df = submission_df.sort_values('id').reset_index(drop=True)

    print(f"Submission shape: {submission_df.shape}")
    print(f"Submission preview:\n{submission_df.head()}")
    print(f"Prediction statistics:")
    print(f"  Min: {submission_df['has_cactus'].min():.6f}")
    print(f"  Max: {submission_df['has_cactus'].max():.6f}")
    print(f"  Mean: {submission_df['has_cactus'].mean():.6f}")

    # Step 7: Save submission file
    submission_path = os.path.join(BASE_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")

    # Also save to output directory
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    output_submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(output_submission_path, index=False)
    print(f"Submission also saved to: {output_submission_path}")

    # Create output info dictionary
    output_info = {
        "submission_file_path": submission_path,
        "output_submission_path": output_submission_path,
        "model_scores": {model_name: float(score) for model_name, score in model_cv_scores.items()},
        "fold_scores": {model_name: [float(s) for s in scores] for model_name, scores in fold_scores.items()},
        "num_folds": cv.get_n_splits(),
        "num_train_samples": len(X),
        "num_test_samples": len(X_test),
        "prediction_min": float(submission_df['has_cactus'].min()),
        "prediction_max": float(submission_df['has_cactus'].max()),
        "prediction_mean": float(submission_df['has_cactus'].mean()),
    }

    print("\nWorkflow completed successfully!")
    return output_info

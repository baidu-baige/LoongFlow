# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import pandas as pd

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
# Import hypothetical components
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/832d0196-b83e-4fa9-8ea2-3588ff903a43/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline for RNA degradation prediction.
    
    Returns:
        dict: A dictionary containing the path to the final submission file.
    """
    # 1. Load the COMPLETE dataset
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy (5-fold Stratified by SN_filter)
    cv = cross_validation(X, y)

    # 3. Initialize containers for Out-of-Fold (OOF) and Test predictions
    # all_test_preds: {model_name: [fold_1_df, fold_2_df, ...]}
    # all_oof_list: {model_name: [fold_1_df, fold_2_df, ...]}
    all_test_preds = {name: [] for name in PREDICTION_ENGINES.keys()}
    all_oof_list = {name: [] for name in PREDICTION_ENGINES.keys()}

    # 4. Execute Cross-Validation Loop
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Split data for the current fold
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Apply graph feature engineering consistently across all sets
        X_train_t, y_train_t, X_val_t, y_val_t, X_test_t = create_features(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )

        # Train and predict for each registered model engine
        for model_name, train_fn in PREDICTION_ENGINES.items():
            # train_fn returns (val_predictions_df, test_predictions_df)
            val_preds_df, test_preds_df = train_fn(
                X_train_t, y_train_t, X_val_t, y_val_t, X_test_t
            )

            all_oof_list[model_name].append(val_preds_df)
            all_test_preds[model_name].append(test_preds_df)

    # 5. Consolidate OOF predictions (concatenating fold results and sorting by original index)
    all_oof_preds = {}
    for model_name in PREDICTION_ENGINES.keys():
        # Concatenate and sort index to align with the original 'y' DataFrame
        consolidated_oof = pd.concat(all_oof_list[model_name]).sort_index()
        all_oof_preds[model_name] = consolidated_oof

    # 6. Ensemble Predictions
    # The ensemble function calculates the arithmetic mean across all folds/models,
    # flattens the result into submission format, and saves it to OUTPUT_DATA_PATH/submission.csv.
    ensemble(all_oof_preds, all_test_preds, y)

    # 7. Prepare and return deliverable info
    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Expected submission file not found at {submission_path}")

    output_info = {
        "submission_file_path": submission_path,
        "model_engines": list(PREDICTION_ENGINES.keys()),
        "num_folds": cv.n_splits
    }

    return output_info

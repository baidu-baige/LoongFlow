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
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/bb8b0571-64fb-49b6-8e49-1fc2d52da49b/2/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This function integrates all pipeline components (data loading, feature engineering, 
    cross-validation, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    """
    # 1. Load the full dataset (production mode)
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Initialize the cross-validation strategy
    splitter = cross_validation(X, y)

    # Prepare data structures to store out-of-fold (OOF) and test predictions
    # This accommodates multiple models if registered in PREDICTION_ENGINES
    all_oof_preds = {name: np.zeros(len(y)) for name in PREDICTION_ENGINES}
    all_test_preds = {name: [] for name in PREDICTION_ENGINES}

    # 3. Execute the Cross-Validation Loop
    for train_index, val_index in splitter.split(X, y):
        # Split features and labels for the current fold
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Apply feature engineering to the fold
        # create_features fits transformers on the training fold and transforms all sets
        X_tr_f, y_tr_f, X_val_f, y_val_f, X_test_f = create_features(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )

        # Train each model and collect internal validation and external test predictions
        for name, train_func in PREDICTION_ENGINES.items():
            val_preds, test_preds = train_func(X_tr_f, y_tr_f, X_val_f, y_val_f, X_test_f)

            # Store OOF predictions to evaluate model performance
            all_oof_preds[name][val_index] = val_preds
            # Store test predictions to be averaged/ensembled later
            all_test_preds[name].append(test_preds)

    # 4. Evaluation and Ensembling
    model_scores = {}
    for name in PREDICTION_ENGINES:
        # Calculate AUC for each model using the collected OOF predictions
        score = roc_auc_score(y, all_oof_preds[name])
        model_scores[name] = float(score)

    # Aggregate predictions across folds and across models
    final_test_preds = ensemble(all_oof_preds, all_test_preds, y)

    # 5. Generate Deliverables and Save Artifacts
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")

    # Format the submission according to sample_submission_null.csv
    # Based on the task description and data properties, the submission should have 
    # the prediction probabilities in the first column, followed by the metadata.
    sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission_null.csv")

    # Load the sample structure
    # If it has a header, pd.read_csv will detect it and we replace the data column.
    sample_sub = pd.read_csv(sample_sub_path)

    # Ensure the prediction array matches the sample length
    # Note: load_data and create_features use the same test.csv, so lengths should align.
    sample_sub.iloc[:, 0] = final_test_preds

    # Save the final submission file preserving the original column structure/headers
    sample_sub.to_csv(submission_file_path, index=False)

    # Return JSON-serializable output summary
    output_info = {
        "submission_file_path": submission_file_path,
        "model_scores": model_scores
    }

    return output_info

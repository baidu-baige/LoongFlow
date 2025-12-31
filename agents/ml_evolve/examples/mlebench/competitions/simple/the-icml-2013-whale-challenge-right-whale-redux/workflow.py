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

# Define the data paths
BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/5872ea81-cdbf-4bc6-8da8-7d0e82d40021/2/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # Step 0: Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Step 1: Load full dataset
    # We must call load_data with validation_mode=False for the production run.
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # Step 2: Set up cross-validation strategy
    # The cross_validation function returns a StratifiedKFold splitter.
    cv_splitter = cross_validation(X, y)

    # Step 3: Initialize storage for Out-of-Fold (OOF) and test predictions
    # We use the provided EfficientNet-B0 engine. 
    # Note: Although instructions mention ResNet18, no implementation or artifact was provided in the context.
    # The ensemble function is designed to handle cases where only EfficientNet is available.
    model_name = "efficientnet_b0"
    train_fn = PREDICTION_ENGINES[model_name]

    oof_preds = pd.Series(np.nan, index=X.index)
    test_preds_list = []

    # Step 4: Execute Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        # Split data for the current fold
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        # a. Generate spectral features for this fold
        # create_features uses GPU-accelerated Log-Mel Spectrograms.
        X_tr_f, y_tr_f, X_va_f, y_va_f, X_te_f = create_features(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )

        # b. Train the primary EfficientNet-B0 model and collect predictions
        # train_fn returns (validation_predictions, test_predictions).
        val_p, test_p = train_fn(
            X_tr_f, y_tr_f, X_va_f, y_va_f, X_te_f
        )

        # c. Store validation results and test set predictions
        # val_p is a pd.Series indexed with X_val_fold.index.
        oof_preds.iloc[val_idx] = val_p.values
        test_preds_list.append(test_p)

    # Step 5: Execute Ensembling
    # Aggregate all model predictions into the format expected by the ensemble component.
    all_oof_preds = {model_name: oof_preds}
    all_test_preds = {model_name: test_preds_list}

    # The ensemble function calculates the final weighted average and saves submission.csv.
    # It identifies model architectures by the keys in the dictionary.
    final_test_preds = ensemble(
        all_oof_preds=all_oof_preds,
        all_test_preds=all_test_preds,
        y_true_full=y
    )

    # Step 6: Generate Deliverables
    # Calculate the overall Cross-Validation AUC score.
    cv_auc = float(roc_auc_score(y, oof_preds))

    # Construct the final output dictionary
    output_info = {
        "submission_file_path": os.path.join(OUTPUT_DATA_PATH, "submission.csv"),
        "model_scores": {
            model_name: cv_auc
        },
        "oof_auc": cv_auc
    }

    return output_info

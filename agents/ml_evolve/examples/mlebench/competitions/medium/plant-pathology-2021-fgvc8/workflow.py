# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import pandas as pd

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/1ae93cb9-976c-4242-a6ab-6fcbc92f5502/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline for Plant Pathology 2021.
    """
    # 1. Load the complete dataset in production mode
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Initialize the cross-validation strategy
    # The cross_validation function returns a StratifiedLabelSplitter
    cv = cross_validation(X, y)

    all_val_probs_list = []
    all_test_preds_list = []

    # 3. Execute the 5-fold cross-validation pipeline
    for train_idx, val_idx in cv.split(X):
        # Split the data for the current fold
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # a. Per-fold feature engineering (augmentations and resizing)
        # Note: X_test is processed repeatedly per fold as per the create_features interface
        X_tr_feat, y_tr_feat, X_val_feat, y_val_feat, X_te_feat = create_features(
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            X_test
        )

        # b. Model Training and Prediction
        # We use the EfficientNet-B4 engine defined in train_and_predict
        train_fn = PREDICTION_ENGINES["efficientnet_b4"]
        val_probs, test_probs = train_fn(
            X_tr_feat,
            y_tr_feat,
            X_val_feat,
            y_val_feat,
            X_te_feat
        )

        # c. Record results with image filenames as index for consistent ensembling
        # The index is mapped from the 'image_path' column to ensure unique identification
        val_probs.index = X_val_fold['image_path'].values
        test_probs.index = X_test['image_path'].values

        all_val_probs_list.append(val_probs)
        all_test_preds_list.append(test_probs)

    # 4. Aggregation and Ensembling
    # Concatenate all out-of-fold predictions to reconstruct the full training set predictions
    all_oof_df = pd.concat(all_val_probs_list)

    # Ensure ground truth labels are indexed by filename for consistency
    y_true_full = y.copy()
    y_true_full.index = X['image_path'].values

    # Perform ensembling (averaging across folds and applying post-processing logic)
    all_oof_preds = {"efficientnet_b4": all_oof_df}
    all_test_preds = {"efficientnet_b4": all_test_preds_list}

    submission_df = ensemble(
        all_oof_preds=all_oof_preds,
        all_test_preds=all_test_preds,
        y_true_full=y_true_full
    )

    # 5. Save final artifacts and deliverables
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # 6. Return metadata for the pipeline execution
    output_info = {
        "submission_file_path": submission_file_path,
        "model_used": "efficientnet_b4",
        "folds": 5,
        "status": "success"
    }

    return output_info

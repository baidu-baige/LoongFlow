# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/7dbf3696-f36d-4f87-8e59-bb45d92869c5/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    1. Loads the full dataset.
    2. Initializes a Stratified 5-fold cross-validation strategy.
    3. Iteratively trains models (EfficientNet-B0) on each fold.
    4. Collects Out-Of-Fold (OOF) and test set predictions.
    5. Ensembles the predictions across all folds and models.
    6. Generates and saves the final submission file.
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Step 1: Load the full dataset
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # Step 2: Define cross-validation strategy
    cv = cross_validation(X, y)

    # Step 3: Initialize placeholders for predictions
    # all_oof_preds maps model name to a full-length array of OOF predictions
    all_oof_preds = {name: np.zeros(len(y)) for name in PREDICTION_ENGINES}
    # all_test_preds maps model name to a list of test predictions (one per fold)
    all_test_preds = {name: [] for name in PREDICTION_ENGINES}

    # Step 4: Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Split data into training and validation sets for the current fold
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Apply feature engineering (metadata standardization)
        X_tr, y_tr, X_v, y_v, X_te = create_features(
            X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
        )

        # Train and predict with each registered model engine
        for model_name, train_func in PREDICTION_ENGINES.items():
            val_preds, test_preds = train_func(X_tr, y_tr, X_v, y_v, X_te)

            # Store OOF predictions in the correct indices
            all_oof_preds[model_name][val_idx] = val_preds.values
            # Store test predictions for later ensembling
            all_test_preds[model_name].append(test_preds)

    # Step 5: Ensemble predictions from all folds and models
    # Note: ensemble() calculates the mean of folds then the mean of models.
    final_test_preds = ensemble(all_oof_preds, all_test_preds, y)

    # Step 6: Calculate CV scores (Log Loss) for documentation
    model_scores = {}
    for model_name, oof_p in all_oof_preds.items():
        # Ensure oof_p is treated as probabilities for Log Loss calculation
        score = log_loss(y, oof_p)
        model_scores[model_name] = float(score)

    # Step 7: Final Submission Generation
    # test_ids and X_test are aligned from load_data
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': final_test_preds
    })

    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # Return JSON-serializable execution summary
    return {
        "submission_file_path": submission_file_path,
        "model_scores": model_scores,
        "status": "success"
    }

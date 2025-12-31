# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/ead961ce-50a1-41ec-89e9-4cc0d527fbe5/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # 1. Load full dataset
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy
    cv = cross_validation(X, y)

    # Initialize containers for OOF and Test predictions
    # Mapping in load_data: 1->0, 2->1, 3->2, 4->3, 6->4, 7->5 (6 classes total)
    num_classes = 6
    all_oof_preds = {name: np.zeros((len(X), num_classes)) for name in PREDICTION_ENGINES}
    all_test_preds = {name: [] for name in PREDICTION_ENGINES}

    # 3. Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Split data for this fold
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Apply feature engineering
        X_t, y_t, X_v, y_v, X_te = create_features(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            X_test
        )

        # Train and predict with each registered model
        for name, train_fn in PREDICTION_ENGINES.items():
            val_probs, test_probs = train_fn(X_t, y_t, X_v, y_v, X_te)

            # Store fold results
            all_oof_preds[name][val_idx] = val_probs
            all_test_preds[name].append(test_probs)

    # 4. Calculate CV Scores for individual models
    model_scores = {}
    for name, oof_probs in all_oof_preds.items():
        oof_labels = np.argmax(oof_probs, axis=1)
        score = accuracy_score(y, oof_labels)
        model_scores[name] = float(score)

    # 5. Ensemble predictions from all models
    # This averages probabilities across all folds/models and maps indices back to original classes
    final_test_predictions = ensemble(all_oof_preds, all_test_preds, y)

    # 6. Generate deliverables
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Create submission file
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'Cover_Type': final_test_predictions
    })

    submission_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    # 7. Collect and return output info
    output_info = {
        "submission_file_path": submission_path,
        "model_scores": model_scores
    }

    return output_info

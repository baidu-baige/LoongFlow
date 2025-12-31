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

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/068f0c14-e630-462e-bc46-9d2d4b1d5fc3/5/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This function integrates all pipeline components (data loading, feature engineering, 
    cross-validation, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    """
    # 1. Load full dataset for production run
    # Requirement: MUST call load_data(validation_mode=False)
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up the cross-validation strategy (Stratified 5-Fold)
    cv = cross_validation(X, y)

    # 3. Initialize prediction containers
    all_oof_preds = {}
    all_test_preds = {}
    model_scores = {}
    num_classes = 5

    # Ensure the output directory exists for artifacts
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Iterate through each model configuration in the prediction engine registry
    for model_name, train_fn in PREDICTION_ENGINES.items():
        # Initialize an array to store full out-of-fold probability predictions
        oof_probs = np.zeros((len(X), num_classes))
        # Store test probability predictions from each fold
        fold_test_preds_list = []

        # Perform cross-validation training and inference
        # Iterates through 5 folds as defined by StratifiedKFold
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Split features and labels into training and validation sets for the current fold
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Apply feature engineering / transformations
            # In this pipeline, create_features persists augmentation settings and returns metadata
            Xt_train, yt_train, Xt_val, yt_val, Xt_test = create_features(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
            )

            # Train the model and perform inference on validation and test sets
            # The train_fn (e.g., train_efficientnet_b4_ns_heavy) handles:
            # - Training for 10 epochs
            # - Best weight selection
            # - 4-view TTA (Original, Horiz Flip, Vert Flip, Transpose)
            val_probs, test_probs = train_fn(
                Xt_train, yt_train, Xt_val, yt_val, Xt_test
            )

            # Record the validation probabilities for OOF assessment
            oof_probs[val_idx] = val_probs
            # Record the test probabilities for ensembling
            fold_test_preds_list.append(test_probs)

        # Store model-specific results for the ensemble stage
        all_oof_preds[model_name] = oof_probs
        all_test_preds[model_name] = fold_test_preds_list

        # Calculate CV Accuracy for logging and validation
        oof_labels = np.argmax(oof_probs, axis=1)
        score = accuracy_score(y.values, oof_labels)
        model_scores[model_name] = float(score)

        # Save model-specific OOF predictions to the artifact directory
        np.save(os.path.join(OUTPUT_DATA_PATH, f"{model_name}_oof.npy"), oof_probs)

    # 4. Ensemble all model and fold predictions using soft-voting
    # The ensemble function averages probabilities across all folds and models
    y_true_full = y.values
    final_test_labels = ensemble(all_oof_preds, all_test_preds, y_true_full)

    # 5. Generate the final submission file
    submission_df = pd.DataFrame({
        'image_id': test_ids,
        'label': final_test_labels
    })
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # 6. Construct and return the workflow summary
    # Must be JSON-serializable
    output_info = {
        "submission_file_path": submission_file_path,
        "model_scores": model_scores,
        "num_folds": 5,
        "num_classes": 5,
        "status": "success"
    }

    return output_info

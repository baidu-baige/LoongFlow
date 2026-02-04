# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux-ml/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/ed330620-ed29-4387-b009-fed5bf45c1a8/11/executor/output"


def workflow() -> dict:
    """
    Executes the complete machine learning workflow to generate the required deliverables.
    Returns a dictionary containing paths to artifacts and performance metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # Start timer
    start_time = time.time()

    # 1. Load data
    print("Loading data...")
    X, y, X_test, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy
    print("Setting up cross-validation...")
    cv = cross_validation(X, y)

    # 3. Initialize storage for predictions
    all_oof_preds = {model_name: np.zeros(len(X)) for model_name in PREDICTION_ENGINES}
    all_test_preds = {model_name: [] for model_name in PREDICTION_ENGINES}
    fold_scores = {model_name: [] for model_name in PREDICTION_ENGINES}

    # 4. OUTER LOOP: for each fold
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        fold_start_time = time.time()
        print(f"\n=== Processing Fold {fold_idx + 1}/{cv.get_n_splits()} ===")

        # Split train/validation data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Call create_features() for this fold (only once per fold)
        print("Creating features...")
        X_train_fe, y_train_fe, X_val_fe = create_features(X_train, y_train, X_val)
        _, _, X_test_fe = create_features(X_train, y_train, X_test)

        # INNER LOOP: for each model
        for model_name, prediction_func in PREDICTION_ENGINES.items():
            print(f"\nTraining {model_name}...")
            model_start_time = time.time()

            # Train and predict
            val_preds, test_preds = prediction_func(
                X_train_fe, y_train_fe,
                X_val_fe, y_val,
                X_test_fe
            )

            # Store OOF predictions
            all_oof_preds[model_name][val_idx] = val_preds

            # Store test predictions (append to list)
            all_test_preds[model_name].append(test_preds)

            # Calculate and store fold score
            fold_score = roc_auc_score(y_val, val_preds)
            fold_scores[model_name].append(fold_score)
            print(f"{model_name} Fold {fold_idx + 1} AUC: {fold_score:.4f} "
                  f"(Time: {time.time() - model_start_time:.1f}s)")

        print(f"Fold {fold_idx + 1} completed in {time.time() - fold_start_time:.1f} seconds")

    # 5. Post-process results
    # Calculate overall OOF scores and standard deviations
    model_scores = {}
    model_score_stds = {}
    for model_name, oof_preds in all_oof_preds.items():
        score = roc_auc_score(y, oof_preds)
        std = np.std(fold_scores[model_name])
        model_scores[model_name] = score
        model_score_stds[model_name] = std
        print(f"\n{model_name} Performance:")
        print(f"  Overall OOF AUC: {score:.4f}")
        print(f"  Fold AUCs: {fold_scores[model_name]}")
        print(f"  Std Dev: {std:.4f}")

    # Ensemble predictions
    print("\nEnsembling predictions...")
    final_test_preds = ensemble(all_oof_preds, all_test_preds, y)

    # 6. Save artifacts
    # Create submission file
    submission_df = pd.DataFrame({
        'request_id': test_ids,
        'requester_received_pizza': final_test_preds
    })
    submission_path = os.path.join(OUTPUT_DATA_PATH, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)

    # Save model predictions and metrics
    predictions_path = os.path.join(OUTPUT_DATA_PATH, 'predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump({
            'model_scores': model_scores,
            'fold_scores': fold_scores,
            'model_score_stds': model_score_stds
        }, f, ensure_ascii=False, indent=2)

    # 7. Create output dictionary
    output_info = {
        'submission_file_path': submission_path,
        'predictions_file_path': predictions_path,
        'model_scores': model_scores,
        'model_score_stds': model_score_stds,
        'fold_scores': fold_scores,
        'best_model': max(model_scores, key=model_scores.get),
        'best_score': max(model_scores.values()),
        'total_runtime_seconds': time.time() - start_time
    }

    print("\nWorkflow completed successfully!")
    print(f"Total runtime: {output_info['total_runtime_seconds']:.1f} seconds")
    print(f"Best model: {output_info['best_model']} (AUC: {output_info['best_score']:.4f})")

    return output_info

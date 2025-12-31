# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from create_features import create_features
from cross_validation import cross_validation
from ensemble import ensemble
# Assume all component functions are available for import
from load_data import load_data
from train_and_predict import PREDICTION_ENGINES

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/43637237-1f49-4750-a868-8602ac177881/1/executor/output"


def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    """
    # 1. Load full dataset
    X, y, X_test_meta, test_ids = load_data(validation_mode=False)

    # 2. Set up cross-validation strategy
    splitter = cross_validation(X, y)

    # Initialize containers for out-of-fold and test predictions
    model_names = list(PREDICTION_ENGINES.keys())
    all_oof_preds = {name: np.zeros(len(y)) for name in model_names}
    all_test_preds = {name: [] for name in model_names}

    # 3. Training Loop across Folds
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        # Split metadata
        X_train_meta, X_val_meta = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # a. Feature Engineering for this fold (uses internal joblib parallelization)
        X_tr, y_tr, X_v, y_v, X_te = create_features(
            X_train_meta, y_train, X_val_meta, y_val, X_test_meta
        )

        # b. Train each model engine
        for name, engine_func in PREDICTION_ENGINES.items():
            val_preds, test_preds = engine_func(X_tr, y_tr, X_v, y_v, X_te)

            # Record OOF predictions and test predictions for the fold
            all_oof_preds[name][val_idx] = val_preds
            all_test_preds[name].append(test_preds)

    # 4. Compute Model Scores (MAE)
    model_scores = {}
    for name in model_names:
        score = mean_absolute_error(y, all_oof_preds[name])
        model_scores[name] = float(score)

    # 5. Ensemble predictions from all models/folds
    # This averages all fold predictions for each model and then across models
    final_test_preds = ensemble(all_oof_preds, all_test_preds, y)

    # Calculate an aggregate OOF for statistics (averaging across models if multiple exist)
    ensemble_oof = np.mean([all_oof_preds[name] for name in model_names], axis=0)

    # 6. Compute prediction statistics
    prediction_stats = {
        "oof": {
            "mean": float(np.mean(ensemble_oof)),
            "std": float(np.std(ensemble_oof)),
            "min": float(np.min(ensemble_oof)),
            "max": float(np.max(ensemble_oof)),
        },
        "test": {
            "mean": float(np.mean(final_test_preds)),
            "std": float(np.std(final_test_preds)),
            "min": float(np.min(final_test_preds)),
            "max": float(np.max(final_test_preds)),
        }
    }

    # 7. Generate deliverables
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'segment_id': test_ids,
        'time_to_eruption': final_test_preds
    })

    # Ensure output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # Save OOF predictions for transparency/reproducibility
    oof_file_path = os.path.join(OUTPUT_DATA_PATH, "oof_predictions.npy")
    np.save(oof_file_path, ensemble_oof)

    # Return task artifacts
    output_info = {
        "submission_file_path": submission_file_path,
        "model_scores": model_scores,
        "prediction_stats": prediction_stats,
        "oof_file_path": oof_file_path
    }

    return output_info

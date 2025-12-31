# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

BASE_DATA_PATH = "/root/workspace/evolux-ml/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/ed330620-ed29-4387-b009-fed5bf45c1a8/11/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using optimized weighted averaging.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (use for optimization/scoring).

    Returns:
        DT: Final predictions for the Test set.
    """
    # Step 1: Preprocess inputs and validate shapes
    if len(all_oof_preds) == 0 or len(all_test_preds) == 0:
        raise ValueError("No predictions provided for ensembling")

    # Convert all predictions to numpy arrays for consistency
    all_oof_preds = {k: np.array(v) for k, v in all_oof_preds.items()}
    all_test_preds = {k: [np.array(p) for p in v] for k, v in all_test_preds.items()}

    # Validate prediction lengths
    test_lengths = {k: len(p[0]) for k, p in all_test_preds.items()}
    if len(set(test_lengths.values())) > 1:
        raise ValueError("All test predictions must have the same length")

    # Step 2: Calculate model performances and select best models
    model_performances = {}
    for model_name, oof_preds in all_oof_preds.items():
        try:
            auc = roc_auc_score(y_true_full, oof_preds)
            model_performances[model_name] = auc
        except ValueError:
            model_performances[model_name] = 0.5  # Default score for failed models

    # Sort models by performance (descending)
    sorted_models = sorted(model_performances.items(), key=lambda x: -x[1])

    # Step 3: Optimize ensemble weights using validation set
    def objective(weights):
        # Weighted average of predictions
        weighted_preds = sum(w * all_oof_preds[m] for w, m in zip(weights, model_performances.keys()))
        # Calculate negative AUC (since we minimize)
        return -roc_auc_score(y_true_full, weighted_preds)

    # Initial weights based on model performance
    initial_weights = np.array([v for k, v in sorted_models])
    initial_weights = initial_weights / initial_weights.sum()

    # Constraints: weights sum to 1, each weight between 0 and 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(model_performances))]

    # Optimize weights
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )

    if not result.success:
        print("Warning: Weight optimization failed, using performance-based weights")
        optimal_weights = initial_weights
    else:
        optimal_weights = result.x

    # Create weight dictionary
    model_weights = {model: weight for model, weight in zip(model_performances.keys(), optimal_weights)}

    # Step 4: Create weighted ensemble predictions
    ensemble_test_preds = None

    for model_name, test_preds in all_test_preds.items():
        # Average predictions across folds for this model
        model_avg_preds = np.mean(test_preds, axis=0)

        # Apply weight to this model's predictions
        weighted_preds = model_avg_preds * model_weights[model_name]

        # Accumulate weighted predictions
        if ensemble_test_preds is None:
            ensemble_test_preds = weighted_preds
        else:
            ensemble_test_preds += weighted_preds

    # Step 5: Post-process predictions
    # Ensure no NaN or infinite values
    ensemble_test_preds = np.nan_to_num(ensemble_test_preds, nan=0.5, posinf=1.0, neginf=0.0)

    # Clip to valid probability range
    ensemble_test_preds = np.clip(ensemble_test_preds, 0, 1)

    # Convert to appropriate output type (match input type)
    if isinstance(all_test_preds[list(all_test_preds.keys())[0]][0], pd.Series):
        ensemble_test_preds = pd.Series(ensemble_test_preds)
    elif isinstance(all_test_preds[list(all_test_preds.keys())[0]][0], pd.DataFrame):
        ensemble_test_preds = pd.DataFrame(ensemble_test_preds)

    return ensemble_test_preds

import numpy as np
import pandas as pd
import optuna
from typing import Dict, List, Any
from sklearn.metrics import log_loss
from scipy.special import expit, logit
import os

# Task-adaptive type definitions
y = pd.Series
Predictions = np.ndarray

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple backbones using Temperature Scaling and 
    Optuna-optimized Weighted Averaging to minimize Log Loss.
    """
    print(f"Ensemble: Combining predictions from {list(all_val_preds.keys())}")

    # Ensure reproducibility
    np.random.seed(42)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    model_names = list(all_val_preds.keys())
    if not model_names:
        raise ValueError("Ensemble received an empty dictionary of predictions.")

    # 1. Post-Hoc Calibration (Temperature Scaling)
    # Calibrate each model's probability distribution to minimize individual Log Loss
    calibrated_val_preds = {}
    calibrated_test_preds = {}
    best_temperatures = {}

    for name in model_names:
        val_p = np.clip(all_val_preds[name], 1e-7, 1 - 1e-7)
        test_p = np.clip(all_test_preds[name], 1e-7, 1 - 1e-7)
        
        # Convert probabilities back to logits for temperature scaling
        val_logits = logit(val_p)
        test_logits = logit(test_p)

        def temp_objective(trial):
            t = trial.suggest_float('t', 0.1, 5.0)
            scaled_p = expit(val_logits / t)
            return log_loss(y_val, scaled_p)

        study = optuna.create_study(direction='minimize')
        study.optimize(temp_objective, n_trials=50)
        
        best_t = study.best_params['t']
        best_temperatures[name] = best_t
        
        calibrated_val_preds[name] = expit(val_logits / best_t)
        calibrated_test_preds[name] = expit(test_logits / best_t)
        
        score = log_loss(y_val, calibrated_val_preds[name])
        print(f"Model {name} - Best Temperature: {best_t:.4f}, Calibrated LogLoss: {score:.6f}")

    # 2. Weighted Average Optimization
    # Optimize weights to minimize ensemble Log Loss
    if len(model_names) > 1:
        def ensemble_objective(trial):
            weights = [trial.suggest_float(f'w_{name}', 0.0, 1.0) for name in model_names]
            norm_weights = np.array(weights) / (sum(weights) + 1e-12)
            
            ensemble_val = np.zeros_like(calibrated_val_preds[model_names[0]])
            for i, name in enumerate(model_names):
                ensemble_val += norm_weights[i] * calibrated_val_preds[name]
            
            return log_loss(y_val, ensemble_val)

        ensemble_study = optuna.create_study(direction='minimize')
        ensemble_study.optimize(ensemble_objective, n_trials=100)
        
        best_weights_raw = [ensemble_study.best_params[f'w_{name}'] for name in model_names]
        best_weights = np.array(best_weights_raw) / sum(best_weights_raw)
        
        final_score = ensemble_study.best_value
        print(f"Optimal Weights: {dict(zip(model_names, best_weights))}")
        print(f"Ensemble LogLoss: {final_score:.6f}")
    else:
        # Single model case
        best_weights = np.array([1.0])
        final_score = log_loss(y_val, calibrated_val_preds[model_names[0]])
        print(f"Single model ensemble LogLoss: {final_score:.6f}")

    # 3. Generate Final Test Predictions
    final_test_preds = np.zeros_like(calibrated_test_preds[model_names[0]])
    for i, name in enumerate(model_names):
        final_test_preds += best_weights[i] * calibrated_test_preds[name]

    # 4. Final Probability Clipping
    # Prevent infinite Log Loss penalties from overconfident errors
    final_test_preds = np.clip(final_test_preds, 0.001, 0.999)

    # Sanity Checks
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble generated NaN or Infinity values.")
    
    if len(final_test_preds) != len(all_test_preds[model_names[0]]):
        raise ValueError("Ensemble output length mismatch with input test predictions.")

    print("Ensemble complete. Returning optimized robust predictions.")
    return final_test_preds
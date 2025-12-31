# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/5872ea81-cdbf-4bc6-8da8-7d0e82d40021/2/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using a weighted average.
    """
    # Step 1: Aggregate fold predictions for each model (Mean across folds)
    model_means = {}
    for model_name, fold_preds in all_test_preds.items():
        # Ensure each fold prediction is converted to a flat numpy array for consistent averaging
        fold_arrays = [np.array(p).flatten() for p in fold_preds]
        model_means[model_name] = np.mean(fold_arrays, axis=0)

    if not model_means:
        raise ValueError("No model predictions available for ensembling.")

    # Step 2: Identify model architectures (EfficientNet-B0 and ResNet18)
    eff_keys = [k for k in model_means.keys() if 'efficientnet' in k.lower()]
    res_keys = [k for k in model_means.keys() if 'resnet' in k.lower()]

    # Extract architecture-specific means
    eff_mean = np.mean([model_means[k] for k in eff_keys], axis=0) if eff_keys else None
    res_mean = np.mean([model_means[k] for k in res_keys], axis=0) if res_keys else None

    # Step 3: Apply weighted average as specified (0.6 Eff + 0.4 Res)
    if eff_mean is not None and res_mean is not None:
        final_preds = 0.6 * eff_mean + 0.4 * res_mean
    elif eff_mean is not None:
        final_preds = eff_mean
    elif res_mean is not None:
        final_preds = res_mean
    else:
        # Fallback: simple average of all available models
        final_preds = np.mean(list(model_means.values()), axis=0)

    # Step 4: Post-processing (Handle Logits and ensure [0, 1] range)
    def ensure_probabilities(p):
        # If values are significantly outside [0, 1], apply sigmoid
        if np.min(p) < -0.01 or np.max(p) > 1.01:
            p = 1 / (1 + np.exp(-p))
        # Handle numerical precision issues and NaNs
        return np.nan_to_num(np.clip(p, 0, 1), nan=0.0, posinf=1.0, neginf=0.0)

    final_preds = ensure_probabilities(final_preds)

    # Step 5: Create submission.csv with clip and probability columns
    try:
        sample_sub_path = os.path.join(BASE_DATA_PATH, "sampleSubmission.csv")
        if os.path.exists(sample_sub_path):
            sample_sub = pd.read_csv(sample_sub_path)
            if len(sample_sub) == len(final_preds):
                submission = pd.DataFrame({
                    'clip': sample_sub['clip'],
                    'probability': final_preds
                })
                os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
                submission.to_csv(os.path.join(OUTPUT_DATA_PATH, "submission.csv"), index=False)
    except Exception as e:
        # Propagation of errors as per requirements
        raise RuntimeError(f"Failed to create submission.csv: {e}")

    # Step 6: Return final test predictions
    return final_preds

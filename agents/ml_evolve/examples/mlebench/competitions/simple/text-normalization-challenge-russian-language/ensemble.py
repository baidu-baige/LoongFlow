# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/text-normalization-challenge-russian-language/prepared/public"
OUTPUT_DATA_PATH = "output/dce1a922-fb4b-4006-9d03-6f53b7ea0718/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models using a multi-strategy approach.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels (use for optimization/scoring).

    Returns:
        DT: Final predictions for the Test set.
    """
    print("Starting ensemble process...")

    # Step 1: Determine test set size from the predictions
    n_test = None
    for model_name, fold_preds in all_test_preds.items():
        if fold_preds and len(fold_preds) > 0:
            first_fold = fold_preds[0]
            if first_fold is not None:
                n_test = len(first_fold)
                print(f"Detected test set size: {n_test} from model {model_name}")
                break

    if n_test is None:
        raise ValueError("Could not determine test set size from predictions")

    # Step 2: Aggregate fold predictions for each model
    print("\nAggregating fold predictions...")
    aggregated_test_preds = {}

    for model_name, fold_preds in all_test_preds.items():
        print(f"  Processing model: {model_name} with {len(fold_preds)} folds")

        if len(fold_preds) == 0:
            continue

        # For text predictions, use voting across folds
        n_samples = len(fold_preds[0])
        aggregated = []

        for i in range(n_samples):
            fold_values = []
            for fold_pred in fold_preds:
                if fold_pred is None:
                    continue
                if isinstance(fold_pred, (pd.Series, pd.DataFrame)):
                    val = fold_pred.iloc[i] if hasattr(fold_pred, 'iloc') else fold_pred[i]
                elif isinstance(fold_pred, np.ndarray):
                    val = fold_pred[i]
                else:
                    val = fold_pred[i]

                # Convert to string, handle None/NaN
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    fold_values.append(str(val))
                else:
                    fold_values.append('')

            # Use majority voting
            if fold_values:
                counter = Counter(fold_values)
                most_common = counter.most_common(1)[0][0]
                aggregated.append(most_common)
            else:
                aggregated.append('')

        aggregated_test_preds[model_name] = np.array(aggregated)
        print(f"    Aggregated {len(aggregated)} predictions")

    # Step 3: Evaluate OOF predictions to determine model reliability
    print("\nEvaluating OOF predictions...")
    model_accuracies = {}

    if y_true_full is not None:
        if isinstance(y_true_full, pd.Series):
            y_true_arr = y_true_full.values
        elif isinstance(y_true_full, pd.DataFrame):
            y_true_arr = y_true_full.values.flatten()
        else:
            y_true_arr = np.array(y_true_full)

        # Convert to string array
        y_true_arr = np.array(
            [str(y) if y is not None and not (isinstance(y, float) and np.isnan(y)) else '' for y in y_true_arr])

        for model_name, oof_preds in all_oof_preds.items():
            if oof_preds is None:
                continue

            if isinstance(oof_preds, pd.Series):
                oof_arr = oof_preds.values
            elif isinstance(oof_preds, pd.DataFrame):
                oof_arr = oof_preds.values.flatten()
            else:
                oof_arr = np.array(oof_preds)

            oof_arr = np.array(
                [str(p) if p is not None and not (isinstance(p, float) and np.isnan(p)) else '' for p in oof_arr])

            # Calculate accuracy
            if len(oof_arr) == len(y_true_arr):
                accuracy = np.mean(oof_arr == y_true_arr)
                model_accuracies[model_name] = accuracy
                print(f"  {model_name}: OOF accuracy = {accuracy:.4f}")

    # Step 4: Build lookup tables from OOF predictions and ground truth
    # This helps refine predictions based on what we learned during CV
    print("\nBuilding refinement lookup from OOF data...")

    oof_lookup = {}  # Maps (model_pred) -> most_common_correct_answer

    if y_true_full is not None and all_oof_preds:
        for model_name, oof_preds in all_oof_preds.items():
            if oof_preds is None:
                continue

            if isinstance(oof_preds, pd.Series):
                oof_arr = oof_preds.values
            elif isinstance(oof_preds, pd.DataFrame):
                oof_arr = oof_preds.values.flatten()
            else:
                oof_arr = np.array(oof_preds)

            # Build correction lookup
            for pred, true in zip(oof_arr, y_true_arr):
                pred_str = str(pred) if pred is not None and not (isinstance(pred, float) and np.isnan(pred)) else ''
                true_str = str(true) if true is not None and not (isinstance(true, float) and np.isnan(true)) else ''

                if pred_str not in oof_lookup:
                    oof_lookup[pred_str] = Counter()
                oof_lookup[pred_str][true_str] += 1

    print(f"  Built OOF lookup with {len(oof_lookup)} entries")

    # Step 5: Apply multi-strategy ensemble
    print("\nApplying multi-strategy ensemble...")

    # Get base predictions from models
    base_predictions = None
    best_model = None

    if aggregated_test_preds:
        # Use the model with highest OOF accuracy, or first available
        if model_accuracies:
            best_model = max(model_accuracies, key=model_accuracies.get)
        else:
            best_model = list(aggregated_test_preds.keys())[0]

        base_predictions = aggregated_test_preds[best_model]
        print(f"  Using base predictions from: {best_model}")

    # Initialize final predictions
    final_predictions = np.empty(n_test, dtype=object)

    # Strategy counters for logging
    strategy_counts = {
        'model_prediction': 0,
        'oof_correction': 0,
        'voting': 0,
        'fallback': 0
    }

    # Apply strategies for each token
    print("  Applying prediction strategies...")

    for i in range(n_test):
        # Strategy 1: Use model prediction directly
        if base_predictions is not None and i < len(base_predictions):
            model_pred = base_predictions[i]
            model_pred_str = str(model_pred) if model_pred is not None and not (
                    isinstance(model_pred, float) and np.isnan(model_pred)) else ''

            # Check if we can refine using OOF lookup
            if model_pred_str in oof_lookup:
                counter = oof_lookup[model_pred_str]
                most_common_true, count = counter.most_common(1)[0]
                total = sum(counter.values())
                consistency = count / total

                # If high consistency, use the correction
                if consistency >= 0.95 and most_common_true:
                    final_predictions[i] = most_common_true
                    strategy_counts['oof_correction'] += 1
                    continue

            # Use model prediction as-is
            if model_pred_str:
                final_predictions[i] = model_pred_str
                strategy_counts['model_prediction'] += 1
                continue

        # Strategy 2: Voting across all models
        if len(aggregated_test_preds) > 1:
            all_preds = []
            for model_name, preds in aggregated_test_preds.items():
                if preds is not None and i < len(preds):
                    pred = preds[i]
                    pred_str = str(pred) if pred is not None and not (
                            isinstance(pred, float) and np.isnan(pred)) else ''
                    if pred_str:
                        all_preds.append(pred_str)

            if all_preds:
                counter = Counter(all_preds)
                most_common = counter.most_common(1)[0][0]
                final_predictions[i] = most_common
                strategy_counts['voting'] += 1
                continue

        # Strategy 3: Fallback - empty string
        final_predictions[i] = ''
        strategy_counts['fallback'] += 1

    print(f"\nStrategy usage:")
    for strategy, count in strategy_counts.items():
        pct = 100 * count / n_test if n_test > 0 else 0
        print(f"  {strategy}: {count:,} ({pct:.2f}%)")

    # Step 6: Post-processing and validation
    print("\nPost-processing predictions...")

    # Ensure no None or NaN values
    for i in range(n_test):
        if final_predictions[i] is None:
            final_predictions[i] = ''
        elif isinstance(final_predictions[i], float) and np.isnan(final_predictions[i]):
            final_predictions[i] = ''

    # Convert to string array
    final_predictions = np.array([str(p) for p in final_predictions])

    # Verify output requirements
    assert len(final_predictions) == n_test, f"Output length mismatch: {len(final_predictions)} vs {n_test}"

    # Check for NaN (shouldn't happen with strings, but verify)
    nan_count = sum(1 for p in final_predictions if p is None or (isinstance(p, float) and np.isnan(p)))
    assert nan_count == 0, f"Output contains {nan_count} NaN values"

    print(f"\nEnsemble complete: {len(final_predictions)} predictions")

    return final_predictions

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_error

# Task-adaptive type definitions
y = pd.Series                # Target vector type (time_to_eruption)
Predictions = np.ndarray     # Model predictions type (array of floats)

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/predict-volcanic-eruptions-ingv-oe/prepared/public"
OUTPUT_DATA_PATH = "output/bdc750a4-f0a3-4926-871d-f9675d7cf1ef/1/executor/output"

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final output using weighted average blending.
    This stage aims to reduce variance and improve generalization for the MAE metric.
    """
    model_names = list(all_test_preds.keys())
    if not model_names:
        raise ValueError("Ensemble received no predictions to combine.")

    print(f"Ensemble: Starting consolidation of {len(model_names)} models: {model_names}")

    # Step 1: Evaluate individual model scores and prediction correlations
    scores = {}
    for name in model_names:
        # Standard MAE evaluation as per competition requirements
        mae = mean_absolute_error(y_val, all_val_preds[name])
        scores[name] = mae
        print(f"Ensemble Evaluation -> Model: {name:20} | Val MAE: {mae:.2f}")

    # Log correlations if multiple models are present to assess diversity
    if len(model_names) > 1:
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                n1, n2 = model_names[i], model_names[j]
                corr = np.corrcoef(all_val_preds[n1], all_val_preds[n2])[0, 1]
                print(f"Ensemble Correlation -> {n1} vs {n2}: {corr:.4f}")

    # Step 2: Apply ensemble strategy (Weighted Average Blending)
    # The Technical Specification suggests [0.5, 0.5] weights initially.
    # We implement a generalized weighted average that defaults to equal weights 
    # unless specific performance-based weighting logic is required.
    
    if len(model_names) == 1:
        print(f"Single model output detected: {model_names[0]}. Proceeding without blending.")
        final_test_preds = all_test_preds[model_names[0]]
    else:
        # Define weights based on Technical Specification [0.5, 0.5]
        # For N models, we use simple averaging (1/N) which generalizes the [0.5, 0.5] requirement.
        weights = np.ones(len(model_names)) / len(model_names)
        
        # Initialize final prediction array with high precision
        first_key = model_names[0]
        final_test_preds = np.zeros_like(all_test_preds[first_key], dtype=np.float64)
        
        for i, name in enumerate(model_names):
            # Accumulate weighted predictions
            final_test_preds += weights[i] * all_test_preds[name].astype(np.float64)
            
        print(f"Successfully blended {len(model_names)} models with weights: {weights}")

    # Step 3: Final validation and cleanup
    # Ensure the output is robust and conforms to downstream requirements
    if np.isnan(final_test_preds).any() or np.isinf(final_test_preds).any():
        raise ValueError("Ensemble produced invalid values (NaN/Inf). Evaluation aborted.")

    # Ensure output shape matches test set size
    assert len(final_test_preds) == len(all_test_preds[model_names[0]]), \
        "Ensemble output size mismatch with input test predictions."

    print("Ensemble: Final prediction vector generated successfully.")
    return final_test_preds
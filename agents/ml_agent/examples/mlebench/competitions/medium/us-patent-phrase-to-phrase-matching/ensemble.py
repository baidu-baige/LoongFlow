import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.stats import pearsonr

# Task-adaptive type definitions
y = pd.Series           # Target vector type (Similarity scores)
Predictions = np.ndarray # Model predictions type (1D array of floats)

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/02d42284-9bf3-4f97-ab6c-7ea839095b54/3/executor/output"

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final output using an arithmetic mean.
    
    This implementation aggregates the test-set predictions from multiple models (e.g., different folds)
    to produce a robust final score, optimizing for the Pearson correlation coefficient.
    """
    print("Execution: ensemble (Stage 5)")

    model_names = list(all_test_preds.keys())
    num_models = len(model_names)
    
    if num_models == 0:
        raise ValueError("Ensemble failed: No test predictions provided in all_test_preds.")

    print(f"Ensembling results from {num_models} models/folds: {model_names}")

    # Step 1: Evaluate individual model performance on validation data
    # This provides diagnostics on how each fold performed relative to the metric.
    for name in model_names:
        if name in all_val_preds:
            v_preds = all_val_preds[name]
            # Ensure y_val alignment. Pearson correlation is only calculated if 
            # the lengths match perfectly (e.g., if all_val_preds contains full OOF).
            if len(v_preds) == len(y_val):
                # Calculate Pearson Correlation Coefficient
                corr, _ = pearsonr(y_val, v_preds)
                print(f"  - Model '{name}' OOF Pearson Correlation: {corr:.4f}")
            else:
                # Typically, in K-fold, v_preds might be fold-specific subsets.
                # We log the length but skip Pearson calculation to avoid misalignment errors.
                pass

    # Step 2: Apply ensemble strategy (Arithmetic Mean)
    # Technical Specification: Reuse parent logic (Arithmetic mean with uniform weighting)
    
    # Identify the output shape from the first model
    first_model_name = model_names[0]
    sample_preds = all_test_preds[first_model_name]
    test_size = len(sample_preds)
    
    # Initialize accumulator array
    final_test_preds = np.zeros(test_size, dtype=np.float32)
    weight = 1.0 / num_models
    
    for name in model_names:
        fold_preds = all_test_preds[name]
        
        # Integrity Check: Ensure consistency in prediction lengths across models
        if len(fold_preds) != test_size:
            raise ValueError(f"Inconsistent test prediction length for model '{name}': "
                             f"expected {test_size}, got {len(fold_preds)}.")
        
        # Accumulate weighted predictions (efficient vector operation)
        final_test_preds += (fold_preds.astype(np.float32) * weight)

    # Step 3: Quality Control and Post-processing
    # Ensure no invalid values (NaN/Inf) which would break submission/evaluation
    if np.isnan(final_test_preds).any():
        raise ValueError("Ensemble process generated NaN values in predictions.")
    
    if np.isinf(final_test_preds).any():
        raise ValueError("Ensemble process generated Infinity values in predictions.")

    # Clip values to the valid [0, 1] range as per the score definition (0.0 to 1.0)
    # This ensures consistency with the target distribution and competition rules.
    final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

    print(f"Ensemble complete. Final test prediction statistics: "
          f"mean={np.mean(final_test_preds):.4f}, min={np.min(final_test_preds):.4f}, max={np.max(final_test_preds):.4f}")
    
    return final_test_preds
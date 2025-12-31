# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/832d0196-b83e-4fa9-8ea2-3588ff903a43/1/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models by calculating the arithmetic mean
    across all folds and models. Returns an unflattened DataFrame where each row
    is a molecule with target columns containing lists of predicted values.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}.
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
        y_true_full: The Ground Truth labels.

    Returns:
        DT: Final predictions for the Test set as a molecule-level DataFrame (one row per id).
    """
    target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    # Aggregate all prediction DataFrames from every model and every fold into a single list
    all_fold_dfs = []
    for model_name in all_test_preds:
        all_fold_dfs.extend(all_test_preds[model_name])

    if not all_fold_dfs:
        raise ValueError("The all_test_preds dictionary is empty or contains no predictions.")

    # Use the first available prediction DataFrame as a template for the result.
    # This ensures we maintain the correct 'id' column and list-based target structure.
    template_df = all_fold_dfs[0]
    res_df = template_df.copy()

    # Step 1 & 2: Compute the simple arithmetic mean for each molecule and target sequence.
    # We iterate through samples to handle sequences of potentially different lengths (e.g., 107 or 130).
    for i in range(len(template_df)):
        for col in target_cols:
            # Collect the sequences (lists or arrays) for this specific sample and column
            # across all models and folds.
            sample_preds = [df[col].iloc[i] for df in all_fold_dfs]

            # Compute the element-wise mean across all fold/model predictions for this sequence
            avg_sequence = np.mean(sample_preds, axis=0)

            # Store the averaged result back into the result DataFrame.
            # .at is used to correctly assign the list/array object to the specific cell.
            res_df.at[i, col] = avg_sequence.tolist()

    # Step 3: Flatten the molecule-level predictions into the competition submission format
    # (one row per id_seqpos) and save to the specified output directory.
    submission_rows = []
    for _, row in res_df.iterrows():
        sample_id = row['id']
        # Sequence length is determined by the length of the predicted target arrays
        seq_len = len(row['reactivity'])
        for pos in range(seq_len):
            sub_row = {'id_seqpos': f"{sample_id}_{pos}"}
            for col in target_cols:
                sub_row[col] = row[col][pos]
            submission_rows.append(sub_row)

    submission_df = pd.DataFrame(submission_rows)

    # Ensure the destination directory exists and save the submission file
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(output_file, index=False)

    # Final quality checks to ensure no NaNs or Infs propagated through the mean calculation
    if submission_df[target_cols].isna().any().any():
        raise ValueError("Ensemble predictions contain NaN values.")
    if np.isinf(submission_df[target_cols].values).any():
        raise ValueError("Ensemble predictions contain Infinity values.")

    # Return the unflattened DataFrame to match the expected record count (1 row per molecule)
    # as required by the pipeline's internal validation against the test set size.
    return res_df

# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/denoising-dirty-documents/prepared/public"
OUTPUT_DATA_PATH = "output/f88466a1-e032-494a-acbe-a8ee4e4d23cf/9/executor/output"

# Type Hints
DT = pd.DataFrame | pd.Series | np.ndarray


def ensemble(
    all_oof_preds: Dict[str, DT],
    all_test_preds: Dict[str, List[DT]],
    y_true_full: DT
) -> DT:
    """
    Ensembles predictions from multiple models via pixel-wise arithmetic mean.

    This function aggregates predictions from all provided folds (and models) to create a robust
    final prediction. It handles variable image sizes by processing image-by-image and uses
    TensorFlow to leverage GPU resources for the averaging and clipping operations.

    Args:
        all_oof_preds: Dictionary of {model_name: oof_predictions}. (Unused for simple mean)
        all_test_preds: Dictionary of {model_name: [pred_fold_1, pred_fold_2, ...]}.
                        Values are expected to be sequences (Series/List) of numpy arrays (H, W, 1).
        y_true_full: The Ground Truth labels. (Unused for simple mean)

    Returns:
        DT: Final predictions for the Test set as a pd.Series of numpy arrays,
            indexed by the original test image IDs.
    """

    # -----------------------------------------------------------------------
    # 1. Collect Prediction Sources
    # -----------------------------------------------------------------------
    # Flatten the dictionary structure to a single list of prediction sets (one per fold per model).
    prediction_sources = []
    if all_test_preds:
        for model_name, folds in all_test_preds.items():
            if folds:
                prediction_sources.extend(folds)

    if not prediction_sources:
        print("Warning: No predictions provided to ensemble. Returning empty Series.")
        return pd.Series([], dtype=object)

    # -----------------------------------------------------------------------
    # 2. Establish Reference Index
    # -----------------------------------------------------------------------
    # We assume all models/folds predict on the same test set in the same order.
    # Use the first prediction set to establish the index (image IDs).
    ref_preds = prediction_sources[0]
    if isinstance(ref_preds, (pd.Series, pd.DataFrame)):
        test_indices = ref_preds.index
    else:
        # Fallback for list/array inputs
        test_indices = pd.RangeIndex(len(ref_preds))

    final_preds_list = []

    # -----------------------------------------------------------------------
    # 3. Iterate Image-by-Image (Handling Variable Sizes)
    # -----------------------------------------------------------------------
    # Since test images have different dimensions (H, W), we cannot batch the entire dataset
    # into a single (N_samples, H, W, C) tensor. We process one image ID at a time.

    for idx in test_indices:
        image_pixels_stack = []

        for preds in prediction_sources:
            try:
                val = None
                # Retrieve the prediction for the current index
                if isinstance(preds, (pd.Series, pd.DataFrame)):
                    # Robust label-based indexing
                    if idx in preds.index:
                        val = preds.loc[idx]

                        # Handle cases where a DataFrame row is returned or data is nested
                        if isinstance(val, pd.Series):
                            if 'data' in val:
                                val = val['data']
                            elif len(val) == 1:
                                val = val.iloc[0]
                        elif isinstance(val, pd.DataFrame):
                            if 'data' in val.columns:
                                val = val['data'].iloc[0]
                else:
                    # List/Array: use positional indexing
                    if isinstance(test_indices, pd.RangeIndex) and test_indices.start == 0 and test_indices.step == 1:
                        pos = idx
                    else:
                        # If indices are IDs but preds is a list, map ID to position
                        # This is a fallback and assumes ordering is preserved
                        try:
                            pos = list(test_indices).index(idx)
                        except ValueError:
                            pos = -1

                    if 0 <= pos < len(preds):
                        val = preds[pos]

                # Validation: Ensure we have a valid numpy array
                if val is not None:
                    if not isinstance(val, np.ndarray):
                        val = np.array(val)

                    # Ensure float32 for TF compatibility
                    if val.dtype != np.float32:
                        val = val.astype(np.float32)

                    image_pixels_stack.append(val)

            except Exception as e:
                print(f"Warning: Issue retrieving prediction for index {idx}: {e}")
                continue

        if not image_pixels_stack:
            # Robustness: If no predictions found for an ID, append None (will likely fail downstream
            # but preserves length) or handle gracefully.
            print(f"Error: No valid predictions found for index {idx}")
            final_preds_list.append(None)
            continue

        # -------------------------------------------------------------------
        # 4. Compute Ensemble (GPU Accelerated)
        # -------------------------------------------------------------------
        # Stack structure: (N_folds, H, W, C)
        # Note: All predictions for the SAME image ID must have the same H, W
        try:
            stack_np = np.stack(image_pixels_stack, axis=0)

            # Execute on GPU if available, else CPU
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                # Convert to Tensor
                stack_tf = tf.convert_to_tensor(stack_np)

                # Compute Arithmetic Mean across the folds (axis 0)
                mean_tf = tf.reduce_mean(stack_tf, axis=0)

                # Clip values to [0, 1] range to ensure valid image intensities
                clipped_tf = tf.clip_by_value(mean_tf, 0.0, 1.0)

                # Convert back to NumPy array
                final_img = clipped_tf.numpy()

            final_preds_list.append(final_img)

        except Exception as e:
            print(f"Error computing ensemble stats for index {idx}: {e}")
            # Fallback: Use the first prediction if ensemble fails
            final_preds_list.append(image_pixels_stack[0])

    # -----------------------------------------------------------------------
    # 5. Construct Final Output
    # -----------------------------------------------------------------------
    # Return a pandas Series indexed by the original test indices/IDs
    return pd.Series(final_preds_list, index=test_indices)

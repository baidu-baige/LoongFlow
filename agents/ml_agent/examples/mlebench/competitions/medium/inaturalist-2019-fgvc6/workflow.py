import os
import numpy as np
import pandas as pd
from typing import Dict

# Import component functions
from load_data import load_data
from preprocess import preprocess
from get_splitter import get_splitter
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/inaturalist-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/57a5d42d-daf8-4241-8197-424dd36c00a6/1/executor/output"

def workflow() -> dict:
    """
    Orchestrates the complete end-to-end machine learning pipeline for the 
    iNaturalist 2019 FGVC6 competition in production mode.
    """
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    # 1. Load the full dataset (production mode)
    train_inputs, train_targets, test_inputs, test_indices = load_data(validation_mode=False)

    # 2. Set up the data splitting strategy
    # The splitter identifies the official validation set or falls back to a 
    # random split for pipeline continuity.
    splitter = get_splitter(train_inputs, train_targets)

    # Containers for outputs from different models/folds
    all_val_outputs: Dict[str, np.ndarray] = {}
    all_test_outputs: Dict[str, np.ndarray] = {}
    model_scores: Dict[str, float] = {}

    # 3. Training and Inference Loop
    # Iterating through the splitter (usually 1 fold for this task)
    for train_idx, val_idx in splitter.split(train_inputs, train_targets):
        # Split data into training and validation sets for this fold
        fold_train_inputs = train_inputs.iloc[train_idx]
        fold_train_targets = train_targets.iloc[train_idx]
        fold_val_inputs = train_inputs.iloc[val_idx]
        fold_val_targets = train_targets.iloc[val_idx]

        # b. Preprocess the datasets into model-ready Dataset objects
        train_ds, train_y, val_ds, val_y, test_ds = preprocess(
            fold_train_inputs, 
            fold_train_targets, 
            fold_val_inputs, 
            fold_val_targets, 
            test_inputs
        )

        # Reconstruct the dense label mapping for metric calculation (must match train_and_predict)
        all_cat_ids = sorted(pd.concat([fold_train_targets, fold_val_targets]).unique())
        cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(all_cat_ids)}
        val_y_mapped = np.array([cat_to_idx[c] for c in fold_val_targets.values])

        # c. Train and Predict using all registered engines
        for model_name, train_func in PREDICTION_ENGINES.items():
            # train_func returns (val_logits, test_logits)
            val_logits, test_logits = train_func(
                train_ds, train_y, val_ds, val_y, test_ds
            )

            # Store predictions for the final ensemble
            all_val_outputs[model_name] = val_logits
            all_test_outputs[model_name] = test_logits

            # Compute CV Score: Top-1 Classification Error
            preds = np.argmax(val_logits, axis=1)
            error_rate = float(np.mean(preds != val_y_mapped))
            model_scores[model_name] = error_rate

        # The current task setup uses a single official validation split
        break

    # 4. Ensemble predictions from all models
    # The ensemble function handles the mapping of dense indices back to competition IDs
    # and formats the output into the required space-separated top-5 string.
    final_test_predictions = ensemble(all_val_outputs, all_test_outputs, fold_val_targets)

    # 5. Compute Prediction Statistics
    def calculate_stats(logits_dict: Dict[str, np.ndarray]) -> dict:
        """Calculates stats based on the max softmax probability of the mean logits."""
        avg_logits = np.mean(list(logits_dict.values()), axis=0)
        # Numerically stable softmax
        shifted = avg_logits - np.max(avg_logits, axis=1, keepdims=True)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        max_probs = np.max(probs, axis=1)
        return {
            "mean": float(np.mean(max_probs)),
            "std": float(np.std(max_probs)),
            "min": float(np.min(max_probs)),
            "max": float(np.max(max_probs)),
        }

    prediction_stats = {
        "oof": calculate_stats(all_val_outputs),
        "test": calculate_stats(all_test_outputs)
    }

    # 6. Generate the final submission file
    submission_df = pd.DataFrame({
        'id': test_indices,
        'predicted': final_test_predictions
    })
    submission_file_path = os.path.join(OUTPUT_DATA_PATH, "submission.csv")
    submission_df.to_csv(submission_file_path, index=False)

    # 7. Collect and return deliverables
    output_info = {
        "submission_file_path": submission_file_path,
        "model_scores": model_scores,
        "prediction_stats": prediction_stats,
    }

    return output_info
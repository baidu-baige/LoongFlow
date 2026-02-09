import os
import numpy as np
from typing import Dict, List, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/tensorflow-speech-recognition-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/fe927f25-451a-41da-a547-cdb392b784d8/1/executor/output"

# Task-adaptive type definitions
y = np.ndarray           # Target vector type (processed integer labels)
Predictions = np.ndarray # Model predictions type (probabilities or final label strings)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models into a final output using soft-voting.

    Args:
        all_val_preds (Dict[str, Predictions]): Dictionary mapping model names to their out-of-fold probability predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model names to their aggregated test probability predictions.
        y_val (y): Ground truth integer labels for the validation set.

    Returns:
        Predictions: Final test set predictions as an array of class label strings.
    """
    print(f"Starting ensemble process for {len(all_test_preds)} models/folds.")

    if not all_test_preds:
        raise ValueError("The all_test_preds dictionary is empty. No predictions available to ensemble.")

    # Step 1: Aggregate test predictions using soft-voting (arithmetic average of softmax probabilities)
    # Each array in all_test_preds is expected to be of shape (N_test, 31)
    test_probs_list = list(all_test_preds.values())
    avg_test_probs = np.mean(test_probs_list, axis=0)

    if np.any(np.isnan(avg_test_probs)) or np.any(np.isinf(avg_test_probs)):
        raise ValueError("Ensemble test probabilities contain invalid values (NaN or Inf).")

    # Step 2: Reconstruct the fixed, alphabetically sorted class list (31 classes)
    # This must strictly match the LabelEncoder mapping used in the preprocess stage.
    # The classes are defined as the 30 word folders plus the 'silence' label.
    train_audio_dir = os.path.join(BASE_DATA_PATH, "train/audio")
    if not os.path.exists(train_audio_dir):
        raise FileNotFoundError(f"Required training audio directory missing: {train_audio_dir}")
        
    word_labels = sorted([
        d for d in os.listdir(train_audio_dir)
        if os.path.isdir(os.path.join(train_audio_dir, d)) and not d.startswith('_')
    ])
    all_classes = sorted(word_labels + ['silence'])
    
    # Technical validation: ensure prediction dimensionality matches the class count
    num_classes = len(all_classes)
    if avg_test_probs.shape[1] != num_classes:
        print(f"Warning: Model output dimension ({avg_test_probs.shape[1]}) does not match expected 31-class count.")
        # We proceed to allow argmax, but if indices go out of bounds, an error will correctly propagate.

    # Step 3: Determine final labels based on argmax of averaged probabilities
    final_test_indices = np.argmax(avg_test_probs, axis=1)
    
    # Map the predicted indices back to class name strings
    try:
        final_test_labels = np.array([all_classes[idx] for idx in final_test_indices])
    except IndexError as e:
        raise RuntimeError(f"Argmax index out of range for class mapping. Predicted index: {final_test_indices.max()}, Class list size: {len(all_classes)}") from e

    # Step 4: Evaluate ensemble performance on validation set for diagnostic purposes
    if all_val_preds and y_val is not None:
        val_probs_list = list(all_val_preds.values())
        avg_val_probs = np.mean(val_probs_list, axis=0)
        final_val_indices = np.argmax(avg_val_probs, axis=1)
        
        # Ensure y_val is in a comparable numpy array format
        y_val_arr = np.array(y_val)
        if y_val_arr.shape == final_val_indices.shape:
            # Multi-class accuracy calculation
            accuracy = np.mean(final_val_indices == y_val_arr)
            print(f"Ensemble Validation Accuracy: {accuracy:.4f}")
        else:
            print(f"Validation shape mismatch: y_val {y_val_arr.shape} vs preds {final_val_indices.shape}")

    print(f"Ensemble complete. Generated predictions for {len(final_test_labels)} test samples.")
    return final_test_labels
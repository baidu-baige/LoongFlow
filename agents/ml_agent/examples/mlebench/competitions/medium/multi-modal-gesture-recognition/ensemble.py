import numpy as np
import scipy.ndimage
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Tuple

# Task-adaptive type definitions
# y_val: List of ground truth gesture sequences, e.g., [[2, 12, 3], [1, 5]]
y = List[List[int]]
# Predictions: Final output format as a list of space-separated strings
Predictions = List[str]
# Probabilities: List of (T, 21) numpy arrays containing frame-level probabilities
Probabilities = List[np.ndarray]

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/multi-modal-gesture-recognition/prepared/public"
OUTPUT_DATA_PATH = "output/7d9b4fa5-39b3-4a58-b088-17228f41073d/11/executor/output"

def levenshtein_dist(a: List[int], b: List[int]) -> int:
    """Computes the Levenshtein distance between two sequences."""
    n, m = len(a), len(b)
    if n > m: a, b = b, a; n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]: change += 1
            current[j] = min(add, delete, change)
    return current[n]

def decode_single(avg_logits: np.ndarray) -> str:
    """
    Decodes a single sequence using logit averaging results, Gaussian smoothing, 
    and duration-based filtering.
    
    Args:
        avg_logits (np.ndarray): (T, 21) matrix of averaged log-probabilities.
        
    Returns:
        str: Space-separated sequence of gesture IDs (1-20).
    """
    # Technical Parameters from Specification
    gaussian_sigma = 1.5
    min_gesture_duration = 15
    background_threshold = 0.5
    
    T = avg_logits.shape[0]
    if T == 0:
        return ""
        
    # 1. Convert Logits to Probabilities (Softmax-like restoration)
    # avg_logits are already the mean of log-probs
    probs = np.exp(avg_logits - np.max(avg_logits, axis=1, keepdims=True))
    probs /= (np.sum(probs, axis=1, keepdims=True) + 1e-12)
    
    # 2. Gaussian Smoothing (Temporal Refinement)
    # Stabilizes frame-level predictions as per specification
    probs_smoothed = scipy.ndimage.gaussian_filter1d(probs, sigma=gaussian_sigma, axis=0)
    
    # 3. Frame-level Labeling with Background Priority
    # Maintaining integrity of background class (0) to prevent merging
    labels = np.zeros(T, dtype=np.int32)
    for t in range(T):
        p_frame = probs_smoothed[t]
        if p_frame[0] > background_threshold:
            labels[t] = 0
        else:
            # Select best gesture among classes 1-20
            labels[t] = np.argmax(p_frame[1:]) + 1
            
    # 4. Sequence Extraction & Duration Filtering
    gestures = []
    if T > 0:
        curr_label = labels[0]
        curr_len = 1
        
        for i in range(1, T):
            if labels[i] == curr_label:
                curr_len += 1
            else:
                # If segment is a valid gesture and meets duration requirement
                if curr_label != 0 and curr_len >= min_gesture_duration:
                    gestures.append(int(curr_label))
                curr_label = labels[i]
                curr_len = 1
        
        # Final segment check
        if curr_label != 0 and curr_len >= min_gesture_duration:
            gestures.append(int(curr_label))
            
    # 5. Collapse Consecutive Identical Gestures
    # Merges consecutive segments of the same gesture ID
    collapsed_seq = []
    if gestures:
        collapsed_seq.append(gestures[0])
        for g in gestures[1:]:
            if g != collapsed_seq[-1]:
                collapsed_seq.append(g)
    
    return " ".join(map(str, collapsed_seq))

def ensemble(
    all_val_preds: Dict[str, Probabilities],
    all_test_preds: Dict[str, Probabilities],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models via logit averaging and temporal smoothing.

    Args:
        all_val_preds (Dict[str, Probabilities]): Dict of model names to probability lists.
        all_test_preds (Dict[str, Probabilities]): Dict of model names to test probability lists.
        y_val (y): Ground truth sequences for validation scoring.

    Returns:
        Predictions: Final list of space-separated gesture strings for test set.
    """
    print(f"Ensembling {len(all_test_preds)} models...")

    def get_avg_logits(preds_dict: Dict[str, Probabilities]) -> Probabilities:
        if not preds_dict:
            return []
        model_names = list(preds_dict.keys())
        num_samples = len(preds_dict[model_names[0]])
        
        avg_logits_list = []
        for i in range(num_samples):
            # Logit Averaging: Mean of Log-Probabilities
            # This balances model confidence more effectively than simple prob averaging
            log_sum = None
            for m in model_names:
                p = np.clip(preds_dict[m][i], 1e-7, 1.0)
                log_p = np.log(p)
                if log_sum is None:
                    log_sum = log_p
                else:
                    log_sum += log_p
            
            avg_logits_list.append(log_sum / len(model_names))
        return avg_logits_list

    # Step 1: Aggregate predictions via Logit Averaging
    avg_val_logits = get_avg_logits(all_val_preds)
    avg_test_logits = get_avg_logits(all_test_preds)

    # Step 2: Parallel Sequence Decoding across 36 cores
    # Gaussian smoothing and duration filtering are CPU-bound and highly parallelizable
    print(f"Parallel decoding {len(avg_val_logits)} validation samples...")
    with ProcessPoolExecutor(max_workers=36) as executor:
        val_sequences = list(executor.map(decode_single, avg_val_logits))
        
    print(f"Parallel decoding {len(avg_test_logits)} test samples...")
    with ProcessPoolExecutor(max_workers=36) as executor:
        test_sequences = list(executor.map(decode_single, avg_test_logits))

    # Step 3: Evaluation of Ensemble Quality using Levenshtein distance
    if y_val and len(val_sequences) == len(y_val):
        total_dist = 0
        total_true_gestures = 0
        for pred_str, target_seq in zip(val_sequences, y_val):
            pred_list = [int(x) for x in pred_str.split()] if pred_str else []
            total_dist += levenshtein_dist(pred_list, target_seq)
            total_true_gestures += len(target_seq)
        
        score = total_dist / total_true_gestures if total_true_gestures > 0 else 0
        print(f"Ensemble Validation Levenshtein Score: {score:.4f}")
    else:
        print("Validation ground truth check skipped or size mismatch.")

    # Step 4: Final Integrity Check
    test_sample_count = len(next(iter(all_test_preds.values())))
    if len(test_sequences) != test_sample_count:
        raise RuntimeError(f"Ensemble size mismatch: {len(test_sequences)} vs {test_sample_count}")

    print(f"Ensemble stage complete. Processed {len(test_sequences)} test samples.")
    return test_sequences
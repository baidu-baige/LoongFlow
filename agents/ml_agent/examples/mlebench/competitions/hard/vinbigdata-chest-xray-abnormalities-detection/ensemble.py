import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from ensemble_boxes import weighted_boxes_fusion

# Task-adaptive type definitions
# y: List of numpy arrays [xmin, ymin, xmax, ymax, class_id]
# Predictions: List of strings in format "class conf xmin ymin xmax ymax ..."
y = List[np.ndarray]
Predictions = List[str]

def _parse_pred_string(pred_str: str) -> Tuple[List[List[float]], List[float], List[int]]:
    """
    Parses a prediction string into boxes, scores, and labels.
    """
    if not pred_str or (isinstance(pred_str, str) and pred_str.strip() == "14 1.0 0 0 1 1"):
        return [], [], []
    
    parts = pred_str.split()
    boxes = []
    scores = []
    labels = []
    
    # Each prediction is 6 elements: class, conf, xmin, ymin, xmax, ymax
    for i in range(0, len(parts), 6):
        try:
            cls = int(parts[i])
            if cls == 14:
                continue
            conf = float(parts[i+1])
            xmin = float(parts[i+2])
            ymin = float(parts[i+3])
            xmax = float(parts[i+4])
            ymax = float(parts[i+5])
            
            # Normalize to [0, 1] for WBF (assuming 1024 target size from preprocess)
            boxes.append([xmin / 1024.0, ymin / 1024.0, xmax / 1024.0, ymax / 1024.0])
            scores.append(conf)
            labels.append(cls)
        except (ValueError, IndexError):
            continue
            
    return boxes, scores, labels

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models/folds using Weighted Boxes Fusion (WBF).
    
    Args:
        all_val_preds: Out-of-fold predictions from multiple models/folds.
        all_test_preds: Test set predictions from multiple models/folds.
        y_val: Ground truth for validation.

    Returns:
        Predictions: Final consolidated test set predictions.
    """
    print("Stage 5: Initializing Ensemble Stage (Weighted Boxes Fusion)...")
    
    model_names = list(all_test_preds.keys())
    if not model_names:
        raise ValueError("No models found in all_test_preds.")
        
    num_samples = len(all_test_preds[model_names[0]])
    num_models = len(model_names)
    
    print(f"Ensembling {num_models} models/folds for {num_samples} samples.")
    
    # Parameters from Technical Specification
    IOU_THR = 0.5
    SKIP_BOX_THR = 0.001
    WEIGHTS = [1.0] * num_models
    
    final_test_preds = []
    
    for idx in range(num_samples):
        boxes_list = []
        scores_list = []
        labels_list = []
        
        # Collect predictions for this image from all models
        for model_name in model_names:
            pred_str = all_test_preds[model_name][idx]
            b, s, l = _parse_pred_string(pred_str)
            
            # WBF requires lists of lists, even if empty per model
            boxes_list.append(b)
            scores_list.append(s)
            labels_list.append(l)
            
        # Check if any model found abnormalities
        has_abnormalities = any(len(b) > 0 for b in boxes_list)
        
        if not has_abnormalities:
            # All models predicted "No finding" or empty
            final_test_preds.append("14 1.0 0 0 1 1")
            continue
            
        # Apply Weighted Boxes Fusion
        # fused_boxes: (N, 4), fused_scores: (N,), fused_labels: (N,)
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, 
            scores_list, 
            labels_list, 
            weights=WEIGHTS, 
            iou_thr=IOU_THR, 
            skip_box_thr=SKIP_BOX_THR
        )
        
        if len(fused_boxes) == 0:
            final_test_preds.append("14 1.0 0 0 1 1")
        else:
            # Scale back to 1024 coordinate system and format string
            res = []
            for i in range(len(fused_boxes)):
                cls = int(fused_labels[i])
                conf = float(fused_scores[i])
                # Clip normalized coordinates and denormalize to target size
                box = np.clip(fused_boxes[i], 0, 1) * 1024.0
                xmin, ymin, xmax, ymax = box
                res.append(f"{cls} {conf:.4f} {int(round(xmin))} {int(round(ymin))} {int(round(xmax))} {int(round(ymax))}")
            
            final_test_preds.append(" ".join(res))
            
    print(f"Ensemble completed. Generated {len(final_test_preds)} final predictions.")
    
    # Validation check for output consistency
    if len(final_test_preds) != num_samples:
        raise RuntimeError(f"Output size mismatch: Expected {num_samples}, got {len(final_test_preds)}")

    return final_test_preds
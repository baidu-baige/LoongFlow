import torch
from torchvision.ops import nms
from typing import Dict, List, Any
import numpy as np

# Task-adaptive type definitions based on upstream components
y = List[Dict[str, Any]]  # From preprocess: [{'tile_labels': [...], 'crop_labels': [...]}, ...]
Predictions = List[str]    # From train_and_predict: ["Unicode X Y ...", ...]

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using Global Non-Maximum Suppression (NMS).
    
    This implementation merges detections from all models and tiles for each image,
    applying a point-based NMS with a 10-pixel radius threshold to handle overlaps.
    """
    print("Starting Ensemble stage...")

    # Validate inputs
    if not all_test_preds:
        print("Warning: No test predictions provided to ensemble.")
        return []

    model_names = list(all_test_preds.keys())
    num_test_images = len(all_test_preds[model_names[0]])
    num_val_images = len(all_val_preds[model_names[0]]) if model_names else 0

    # Step 1: Evaluate individual model statistics
    # Provides visibility into the density and consistency of model outputs
    for m_name in model_names:
        val_preds = all_val_preds.get(m_name, [])
        val_det_count = sum(len(p.split()) // 3 for p in val_preds)
        avg_det = val_det_count / max(1, len(val_preds))
        print(f"Model '{m_name}': Total Val Detections={val_det_count}, Avg/Image={avg_det:.2f}")

    # Step 2: Apply ensemble strategy (Global NMS)
    # We use a 10px threshold. To implement this with torchvision.ops.nms, 
    # we create 10x10 boxes centered at each point and use iou_threshold=0.
    RADIUS = 10
    HALF_RADIUS = RADIUS / 2.0
    MAX_PREDS_PER_PAGE = 1200
    
    final_test_preds = []

    # Process images sequentially to maximize GPU utilization for NMS without overhead
    for img_idx in range(num_test_images):
        combined_labels = []
        combined_boxes = []
        combined_scores = []
        
        # Aggregate detections from all models
        for m_name in model_names:
            pred_str = all_test_preds[m_name][img_idx]
            if not pred_str or not isinstance(pred_str, str):
                continue
                
            parts = pred_str.split()
            for i in range(0, len(parts), 3):
                try:
                    label = parts[i]
                    x = float(parts[i+1])
                    y = float(parts[i+2])
                    
                    combined_labels.append(label)
                    # Create a box such that overlap occurs if distance < RADIUS
                    combined_boxes.append([
                        x - HALF_RADIUS, 
                        y - HALF_RADIUS, 
                        x + HALF_RADIUS, 
                        y + HALF_RADIUS
                    ])
                    # Since we lack confidence scores, we use 1.0. 
                    # If multiple models are used, NMS will preserve the first model's detection.
                    combined_scores.append(1.0)
                except (ValueError, IndexError):
                    continue
        
        if not combined_boxes:
            final_test_preds.append("")
            continue
            
        # Perform NMS on GPU
        # 1200 points per image is lightweight; moving to GPU is efficient for batch consistency
        boxes_t = torch.tensor(combined_boxes, dtype=torch.float32).cuda()
        scores_t = torch.tensor(combined_scores, dtype=torch.float32).cuda()
        
        # iou_threshold=0.0 ensures any overlapping boxes (distance < 10px) are suppressed
        keep_indices = nms(boxes_t, scores_t, iou_threshold=0.0)
        
        # Enforce competition limit: Do not make more than 1,200 predictions per page
        keep_indices = keep_indices[:MAX_PREDS_PER_PAGE]
        
        # Reconstruct output string
        keep_indices_np = keep_indices.cpu().numpy()
        image_results = []
        for idx in keep_indices_np:
            lbl = combined_labels[idx]
            # Center of the box corresponds to original X, Y
            box = combined_boxes[idx]
            x_orig = int(round(box[0] + HALF_RADIUS))
            y_orig = int(round(box[1] + HALF_RADIUS))
            image_results.append(f"{lbl} {x_orig} {y_orig}")
            
        final_test_preds.append(" ".join(image_results))

    # Step 3: Return final test predictions
    print(f"Ensemble complete. Generated predictions for {len(final_test_preds)} images.")
    
    # Final validation of output alignment
    if len(final_test_preds) != num_test_images:
        raise ValueError(f"Output size mismatch: {len(final_test_preds)} vs {num_test_images}")
        
    return final_test_preds
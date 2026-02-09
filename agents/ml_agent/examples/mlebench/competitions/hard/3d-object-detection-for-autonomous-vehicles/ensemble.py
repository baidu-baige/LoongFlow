import numpy as np
import multiprocessing as mp
from typing import Dict, List, Any, Tuple
from shapely.geometry import Polygon
import gc

# Task-adaptive type definitions
Predictions = List[List[Dict[str, Any]]]
y = List[List[Dict[str, Any]]]

def calculate_3d_iou(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """
    Calculates 3D Intersection over Union (IoU) for two boxes.
    The IoU is calculated as (XY_intersection * Z_intersection) / Union_Volume.
    """
    # 1. Height (Z) overlap
    z1_min, z1_max = box1['center_z'] - box1['height'] / 2, box1['center_z'] + box1['height'] / 2
    z2_min, z2_max = box2['center_z'] - box2['height'] / 2, box2['center_z'] + box2['height'] / 2
    
    inter_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    if inter_z <= 0:
        return 0.0

    # 2. XY-plane (Rotated) overlap
    def get_corners(b):
        x, y, w, l, yaw = b['center_x'], b['center_y'], b['width'], b['length'], b['yaw']
        # Orientation: length (l) is along the front-back axis (yaw direction)
        # width (w) is along the side-to-side axis (yaw + pi/2 direction)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        # Local corners: (front/back dist, left/right dist)
        # Front-Right, Front-Left, Back-Left, Back-Right
        loc_corners = np.array([
            [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]
        ])
        
        # Rotate and translate
        world_corners = np.zeros_like(loc_corners)
        world_corners[:, 0] = x + loc_corners[:, 0] * cos_y - loc_corners[:, 1] * sin_y
        world_corners[:, 1] = y + loc_corners[:, 0] * sin_y + loc_corners[:, 1] * cos_y
        return world_corners

    try:
        poly1 = Polygon(get_corners(box1))
        poly2 = Polygon(get_corners(box2))
        
        if not poly1.is_valid: poly1 = poly1.buffer(0)
        if not poly2.is_valid: poly2 = poly2.buffer(0)
        
        inter_area = poly1.intersection(poly2).area
    except Exception:
        # Fallback for geometric edge cases
        return 0.0

    if inter_area <= 0:
        return 0.0

    # 3. 3D IoU calculation
    inter_vol = inter_area * inter_z
    vol1 = box1['width'] * box1['length'] * box1['height']
    vol2 = box2['width'] * box2['length'] * box2['height']
    union_vol = vol1 + vol2 - inter_vol
    
    return inter_vol / max(union_vol, 1e-8)

def nms_3d_worker(args: Tuple[List[Dict[str, Any]], float, float]) -> List[Dict[str, Any]]:
    """
    Worker function to perform 3D NMS on a single sample's combined predictions.
    """
    boxes, iou_threshold, score_threshold = args
    
    if not boxes:
        return []

    # First pass: Filter by score threshold
    boxes = [b for b in boxes if b['confidence'] >= score_threshold]
    if not boxes:
        return []

    # Sort boxes by confidence descending
    boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    # Class-specific NMS
    classes = set(b['class_name'] for b in boxes)
    
    for cls in classes:
        cls_boxes = [b for b in boxes if b['class_name'] == cls]
        cls_keep_indices = []
        
        indices = np.arange(len(cls_boxes))
        while len(indices) > 0:
            current_idx = indices[0]
            cls_keep_indices.append(current_idx)
            
            if len(indices) == 1:
                break
            
            # Compare current box with remaining boxes
            current_box = cls_boxes[current_idx]
            remaining_indices = indices[1:]
            
            # Optimization: check distance before complex IoU
            ious = []
            for idx in remaining_indices:
                other_box = cls_boxes[idx]
                
                # Coarse distance check (Euclidean distance between centers)
                dist_sq = (current_box['center_x'] - other_box['center_x'])**2 + \
                          (current_box['center_y'] - other_box['center_y'])**2
                
                max_dim = max(current_box['length'], current_box['width'], 
                              other_box['length'], other_box['width'])
                
                # If centers are further apart than twice the largest dimension, 
                # they likely don't overlap significantly. (Conservative check)
                if dist_sq > (max_dim * 2)**2:
                    ious.append(0.0)
                else:
                    ious.append(calculate_3d_iou(current_box, other_box))
            
            ious = np.array(ious)
            filtered_mask = ious <= iou_threshold
            indices = remaining_indices[filtered_mask]
            
        for idx in cls_keep_indices:
            keep.append(cls_boxes[idx])
            
    return sorted(keep, key=lambda x: x['confidence'], reverse=True)

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models using 3D Non-Maximum Suppression (NMS).
    """
    print(f"Ensembling {len(all_test_preds)} models...")
    
    # Configuration
    IOU_THRESHOLD = 0.15 # Mid-range of technical spec (0.1 to 0.2)
    SCORE_THRESHOLD = 0.05 
    
    model_names = list(all_test_preds.keys())
    num_samples = len(all_test_preds[model_names[0]])
    
    # 1. Aggregate validation metrics (Optional Informative Step)
    if all_val_preds and y_val:
        for model_name, val_preds in all_val_preds.items():
            avg_boxes = np.mean([len(p) for p in val_preds])
            print(f"Model '{model_name}' - Avg predictions per validation sample: {avg_boxes:.2f}")

    # 2. Prepare tasks for parallel processing
    ensemble_tasks = []
    print(f"Preparing ensemble tasks for {num_samples} test samples...")
    for i in range(num_samples):
        combined_boxes = []
        for model_name in model_names:
            combined_boxes.extend(all_test_preds[model_name][i])
        ensemble_tasks.append((combined_boxes, IOU_THRESHOLD, SCORE_THRESHOLD))

    # 3. Execute 3D-NMS in parallel
    print(f"Running 3D-NMS on 36 CPU cores...")
    with mp.Pool(processes=36) as pool:
        final_test_preds = pool.map(nms_3d_worker, ensemble_tasks)

    # 4. Final Verification
    assert len(final_test_preds) == num_samples, "Sample count mismatch in ensemble output."
    
    total_boxes_before = sum(len(task[0]) for task in ensemble_tasks)
    total_boxes_after = sum(len(p) for p in final_test_preds)
    print(f"Ensemble complete. Reduced total detections from {total_boxes_before} to {total_boxes_after}.")

    # Clean up
    del ensemble_tasks
    gc.collect()

    return final_test_preds
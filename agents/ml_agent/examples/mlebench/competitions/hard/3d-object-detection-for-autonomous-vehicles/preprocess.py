import numpy as np
import multiprocessing as mp
import gc
import os
from typing import Tuple, List, Dict, Any
from collections import Counter

# Task-adaptive type definitions
X = List[Dict[str, Any]]  # List of metadata dictionaries (processed)
y = List[List[Dict[str, Any]]] # List of lists of ground truth boxes

def quat_to_rot_matrix(q: List[float]) -> np.ndarray:
    """Converts a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def transform_points_to_ego(points: np.ndarray, calibrated: Dict[str, Any]) -> np.ndarray:
    """Transforms lidar points from sensor frame to ego frame."""
    rot = quat_to_rot_matrix(calibrated['rotation'])
    trans = np.array(calibrated['translation'])
    points[:, :3] = points[:, :3] @ rot.T + trans
    return points

def apply_augmentation(points: np.ndarray, boxes: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Applies global rotation, scaling, and translation to points and boxes."""
    # 1. Global Rotation ([-0.785, 0.785] rad)
    noise_rotation = np.random.uniform(-0.785, 0.785)
    rot_sin = np.sin(noise_rotation)
    rot_cos = np.cos(noise_rotation)
    rot_mat = np.array([
        [rot_cos, -rot_sin, 0],
        [rot_sin, rot_cos, 0],
        [0, 0, 1]
    ])
    
    # 2. Global Scaling ([0.95, 1.05])
    noise_scale = np.random.uniform(0.95, 1.05)
    
    # 3. Global Translation (std=0.5)
    noise_trans = np.random.normal(0, 0.5, size=3)
    
    # Transform points
    points[:, :3] = points[:, :3] @ rot_mat.T * noise_scale + noise_trans
    
    # Transform boxes
    augmented_boxes = []
    for box in boxes:
        center = np.array([box['center_x'], box['center_y'], box['center_z']])
        center = center @ rot_mat.T * noise_scale + noise_trans
        
        # New yaw
        yaw = box['yaw'] + noise_rotation
        # Normalize to [-pi, pi]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        
        # New dimensions
        w, l, h = box['width'] * noise_scale, box['length'] * noise_scale, box['height'] * noise_scale
        
        aug_box = {
            **box,
            'center_x': float(center[0]),
            'center_y': float(center[1]),
            'center_z': float(center[2]),
            'yaw': float(yaw),
            'width': float(w),
            'length': float(l),
            'height': float(h)
        }
        augmented_boxes.append(aug_box)
        
    return points, augmented_boxes

def voxelize(points: np.ndarray, voxel_size: List[float], pc_range: List[float], 
             max_points: int = 20, max_voxels: int = 30000) -> Dict[str, np.ndarray]:
    """Voxelizes point cloud into PointPillars-style format."""
    # Filter points out of range
    mask = (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) & \
           (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) & \
           (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    points = points[mask]
    
    if len(points) == 0:
        return {
            'voxels': np.zeros((1, max_points, points.shape[1]), dtype=np.float32),
            'coords': np.zeros((1, 3), dtype=np.int32),
            'num_points': np.zeros(1, dtype=np.int32)
        }
    
    # Compute voxel coordinates
    coords = ((points[:, :3] - np.array(pc_range[:3])) / np.array(voxel_size)).astype(np.int32)
    # Reverse to (z, y, x) for standard voxel encoders
    coords = coords[:, [2, 1, 0]]
    
    # Group points by voxel coordinate
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
    
    num_voxels = min(len(unique_coords), max_voxels)
    voxels = np.zeros((num_voxels, max_points, points.shape[1]), dtype=np.float32)
    voxel_coords = unique_coords[:num_voxels]
    voxel_num_points = np.zeros(num_voxels, dtype=np.int32)
    
    # Fast grouping using sorting
    sort_idx = np.argsort(inverse_indices)
    points_sorted = points[sort_idx]
    inv_sorted = inverse_indices[sort_idx]
    
    # Find start/end of each voxel group
    diff = np.diff(inv_sorted)
    boundaries = np.concatenate([[0], np.where(diff > 0)[0] + 1, [len(inv_sorted)]])
    
    for i in range(num_voxels):
        start, end = boundaries[i], boundaries[i+1]
        n = min(end - start, max_points)
        voxels[i, :n] = points_sorted[start:start+n]
        voxel_num_points[i] = n
        
    return {
        'voxels': voxels,
        'coords': voxel_coords,
        'num_points': voxel_num_points
    }

def process_sample(args: Tuple[Dict[str, Any], List[Dict[str, Any]], bool]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Worker function for parallel processing."""
    meta, boxes, is_train = args
    lidar_meta = meta['lidar']
    
    # Load lidar points
    points = np.fromfile(lidar_meta['filename'], dtype=np.float32)
    # Lyft lidar is typically (x, y, z, intensity, ring_index)
    if len(points) % 5 == 0:
        points = points.reshape(-1, 5)[:, :4]
    else:
        points = points.reshape(-1, 4)
    
    # Transform sensor frame -> ego frame
    points = transform_points_to_ego(points, lidar_meta['calibrated'])
    
    # Augmentation
    if is_train:
        points, boxes = apply_augmentation(points, boxes)
    
    # Voxelization
    voxel_data = voxelize(
        points, 
        voxel_size=[0.2, 0.2, 8.0], 
        pc_range=[-100.0, -100.0, -5.0, 100.0, 100.0, 3.0]
    )
    
    processed_x = {**voxel_data, 'sample_token': meta['sample_token']}
    return processed_x, boxes

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready BEV format using voxelization and CBGS.
    """
    print(f"Starting preprocessing: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples.")
    
    # 1. Class-balanced Grouping and Sampling (CBGS) for training set
    print("Calculating class frequencies for CBGS...")
    all_classes = []
    for boxes in y_train:
        all_classes.extend([b['class_name'] for b in boxes])
    class_counts = Counter(all_classes)
    
    # Define target frequency for each class to calculate repeat factor
    target_freq = 20000 
    class_repeats = {cls: max(1, int(target_freq / count)) for cls, count in class_counts.items()}
    
    train_indices = []
    for i, boxes in enumerate(y_train):
        if not boxes:
            train_indices.append(i)
            continue
        # Max repeat factor among classes in the sample, capped to avoid explosion
        r = max([class_repeats.get(b['class_name'], 1) for b in boxes])
        r = min(r, 4) 
        train_indices.extend([i] * r)
    
    print(f"CBGS: Training samples increased from {len(X_train)} to {len(train_indices)}.")

    # 2. Parallel Processing
    with mp.Pool(36) as pool:
        # Process Test
        print("Processing test set...")
        test_args = [(X_test[i], [], False) for i in range(len(X_test))]
        results_test = pool.map(process_sample, test_args)
        X_test_processed = [r[0] for r in results_test]
        
        # Process Val
        print("Processing validation set...")
        val_args = [(X_val[i], y_val[i], False) for i in range(len(X_val))]
        results_val = pool.map(process_sample, val_args)
        X_val_processed = [r[0] for r in results_val]
        y_val_processed = [r[1] for r in results_val]
        
        # Process Train (Augmented & Balanced)
        print("Processing training set (with augmentations)...")
        train_args = [(X_train[idx], y_train[idx], True) for idx in train_indices]
        results_train = pool.map(process_sample, train_args)
        X_train_processed = [r[0] for r in results_train]
        y_train_processed = [r[1] for r in results_train]

    # Cleanup
    gc.collect()
    
    print("Preprocessing complete.")
    assert len(X_train_processed) == len(y_train_processed)
    assert len(X_val_processed) == len(y_val_processed)
    assert len(X_test_processed) == len(X_test)
    
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed
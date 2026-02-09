import os
import json
import pickle
import pandas as pd
import numpy as np
import gc
from typing import Tuple, Any, List, Dict
from collections import defaultdict

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/3d-object-detection-for-autonomous-vehicles/prepared/public"
OUTPUT_DATA_PATH = "output/341879f4-a7a1-4d18-a801-6d16eb36af1b/1/executor/output"

# Type definitions for this task
X = List[Dict[str, Any]]  # List of metadata dictionaries for each sample
y = List[List[Dict[str, Any]]] # List of lists of ground truth boxes in ego frame
Ids = List[str] # List of sample tokens

def quat_to_rot(q: List[float]) -> np.ndarray:
    """Converts a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_yaw(q: List[float]) -> float:
    """Extracts yaw (rotation around Z-axis) from a quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def transform_box_to_ego(box: Dict[str, Any], ego_pose: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms a box from world coordinates to ego coordinates."""
    # Translation
    dx = box['center_x'] - ego_pose['translation'][0]
    dy = box['center_y'] - ego_pose['translation'][1]
    dz = box['center_z'] - ego_pose['translation'][2]
    
    # Rotation (Inverse of ego rotation: P_ego = R^T * (P_world - T))
    rot_mat = quat_to_rot(ego_pose['rotation'])
    rel_pos = np.array([dx, dy, dz])
    ego_pos = rot_mat.T @ rel_pos
    
    # Yaw transformation
    ego_yaw = get_yaw(ego_pose['rotation'])
    box_yaw_ego = box['yaw'] - ego_yaw
    # Normalize to [-pi, pi]
    box_yaw_ego = (box_yaw_ego + np.pi) % (2 * np.pi) - np.pi
    
    return {
        'center_x': float(ego_pos[0]),
        'center_y': float(ego_pos[1]),
        'center_z': float(ego_pos[2]),
        'width': float(box['width']),
        'length': float(box['length']),
        'height': float(box['height']),
        'yaw': float(box_yaw_ego),
        'class_name': box['class_name']
    }

def parse_prediction_string(prediction_string: str) -> List[Dict[str, Any]]:
    """Parses the space-delimited annotation string from train.csv."""
    if not prediction_string or pd.isna(prediction_string):
        return []
    parts = prediction_string.split()
    objects = []
    # Format: center_x center_y center_z width length height yaw class_name
    for i in range(0, len(parts), 8):
        try:
            obj = {
                'center_x': float(parts[i]),
                'center_y': float(parts[i+1]),
                'center_z': float(parts[i+2]),
                'width': float(parts[i+3]),
                'length': float(parts[i+4]),
                'height': float(parts[i+5]),
                'yaw': float(parts[i+6]),
                'class_name': parts[i+7]
            }
            objects.append(obj)
        except (IndexError, ValueError):
            continue
    return objects

def build_sensor_registry(base_path: str, mode: str) -> Dict[str, Dict[str, Any]]:
    """Builds a lookup table mapping sample_token to its sensor file paths and poses."""
    data_dir = os.path.join(base_path, f'{mode}_data')
    print(f"Building registry for {mode} data from {data_dir}...")
    
    def load_json(name):
        path = os.path.join(data_dir, f'{name}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []

    # Load necessary tables
    sample_data_list = load_json('sample_data')
    ego_poses = {item['token']: item for item in load_json('ego_pose')}
    calibrated = {item['token']: item for item in load_json('calibrated_sensor')}
    sensors = {item['token']: item for item in load_json('sensor')}
    
    # Map sample_token -> list of sample_data entries
    sample_to_data = defaultdict(list)
    for sd in sample_data_list:
        sample_to_data[sd['sample_token']].append(sd)
    
    registry = {}
    for sample_token, data_list in sample_to_data.items():
        entry = {
            'sample_token': sample_token,
            'lidar': None,
            'cameras': {}
        }
        
        for sd in data_list:
            cal_token = sd['calibrated_sensor_token']
            cal_info = calibrated.get(cal_token)
            sensor_token = cal_info['sensor_token'] if cal_info else None
            sensor_info = sensors.get(sensor_token)
            channel = sensor_info['channel'] if sensor_info else 'UNKNOWN'
            
            # Resolve physical file path
            fname = sd['filename']
            # The filenames in JSON use relative paths like "lidar/xxx.bin"
            # We must map them to the actual flat directory structure provided
            base_fname = os.path.basename(fname)
            if 'lidar/' in fname:
                full_path = os.path.join(base_path, f'{mode}_lidar', base_fname)
            elif 'images/' in fname:
                full_path = os.path.join(base_path, f'{mode}_images', base_fname)
            elif 'maps/' in fname:
                full_path = os.path.join(base_path, f'{mode}_maps', base_fname)
            else:
                full_path = os.path.join(base_path, fname)
            
            sd_meta = {
                'filename': full_path,
                'ego_pose': ego_poses.get(sd['ego_pose_token']),
                'calibrated': cal_info,
                'timestamp': sd['timestamp']
            }
            
            if 'LIDAR' in channel:
                entry['lidar'] = sd_meta
            else:
                entry['cameras'][channel] = sd_meta
        
        # We only care about samples that have lidar data for this task
        if entry['lidar']:
            registry[sample_token] = entry
            
    del sample_data_list, ego_poses, calibrated, sensors
    gc.collect()
    return registry

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Lyft 3D dataset registry and annotations.
    """
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    cache_path = os.path.join(OUTPUT_DATA_PATH, "data_registry.pkl")
    
    if os.path.exists(cache_path):
        print(f"Loading cached registry from {cache_path}...")
        with open(cache_path, 'rb') as f:
            full_registry = pickle.load(f)
    else:
        print("Parsing raw JSON data to build registry...")
        train_registry = build_sensor_registry(BASE_DATA_PATH, 'train')
        test_registry = build_sensor_registry(BASE_DATA_PATH, 'test')
        full_registry = {**train_registry, **test_registry}
        with open(cache_path, 'wb') as f:
            pickle.dump(full_registry, f)
        print(f"Registry cached to {cache_path}")

    # Load labels
    train_df = pd.read_csv(os.path.join(BASE_DATA_PATH, "train.csv"))
    test_ids_df = pd.read_csv(os.path.join(BASE_DATA_PATH, "sample_submission.csv"))

    if validation_mode:
        print("Validation mode: Subsetting data to 200 samples.")
        train_df = train_df.head(200)
        test_ids_df = test_ids_df.head(200)

    X_train, y_train = [], []
    print("Preparing training features and ego-normalized targets...")
    for _, row in train_df.iterrows():
        token = row['Id']
        if token in full_registry:
            meta = full_registry[token]
            # Get lidar ego pose as the reference frame
            ref_ego_pose = meta['lidar']['ego_pose']
            
            # Parse world-space boxes and transform to ego-space
            raw_boxes = parse_prediction_string(row['PredictionString'])
            ego_boxes = [transform_box_to_ego(b, ref_ego_pose) for b in raw_boxes]
            
            X_train.append(meta)
            y_train.append(ego_boxes)
            
    X_test, test_ids = [], []
    print("Preparing test features...")
    for _, row in test_ids_df.iterrows():
        token = row['Id']
        if token in full_registry:
            X_test.append(full_registry[token])
            test_ids.append(token)
        else:
            # Should not happen based on competition rules, but handle gracefully
            print(f"Warning: Test token {token} not found in registry.")

    print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples.")
    
    # Final consistency check
    assert len(X_train) == len(y_train), "Train features and targets must align."
    assert len(X_test) == len(test_ids), "Test features and IDs must align."
    
    return X_train, y_train, X_test, test_ids
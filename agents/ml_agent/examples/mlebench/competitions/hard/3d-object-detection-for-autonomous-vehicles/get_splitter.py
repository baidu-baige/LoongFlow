import os
import json
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Any, Iterator, Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/3d-object-detection-for-autonomous-vehicles/prepared/public"
OUTPUT_DATA_PATH = "output/341879f4-a7a1-4d18-a801-6d16eb36af1b/1/executor/output"

# Task-adaptive type definitions
X = List[Dict[str, Any]]  # List of metadata dictionaries for each sample
y = List[List[Dict[str, Any]]] # List of lists of ground truth boxes in ego frame

class SceneBasedGroupKFold:
    """
    Implementation of GroupKFold that supports group-level shuffling to prevent temporal leakage.
    Ensures all frames belonging to the same scene are contained within a single fold.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("Groups (scene_tokens) must be provided for scene-based splitting.")
        
        # Identify unique scene identifiers
        unique_groups = np.unique(groups)
        
        # Use KFold to partition the unique scenes
        group_kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        for train_group_idx, val_group_idx in group_kf.split(unique_groups):
            train_groups = unique_groups[train_group_idx]
            
            # Map scene-level partitions back to individual sample indices
            train_mask = np.isin(groups, train_groups)
            
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(~train_mask)[0]
            
            yield train_idx, val_idx

class SplitterWrapper:
    """
    A wrapper to align the splitter with standard split(X, y) calls while maintaining the 
    internal scene groups logic required for this temporal dataset.
    """
    def __init__(self, splitter: SceneBasedGroupKFold, groups: np.ndarray):
        self.splitter = splitter
        self.groups = groups

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.splitter.get_n_splits()

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # Use the pre-computed scene groups extracted from metadata
        return self.splitter.split(X, y, groups=self.groups)

def get_splitter(X: X, y: y) -> Any:
    """
    Defines a scene-based validation strategy to prevent temporal leakage.
    
    Args:
        X (X): The training features (metadata registry).
        y (y): The training targets (ego-frame boxes).

    Returns:
        Any: A splitter object implementing split() and get_n_splits().
    """
    # Load sample metadata to map sample_token to scene_token
    # This is necessary because scene_token is the unit of temporal grouping
    sample_json_path = os.path.join(BASE_DATA_PATH, "train_data", "sample.json")
    if not os.path.exists(sample_json_path):
        raise FileNotFoundError(f"Required metadata file {sample_json_path} not found.")

    with open(sample_json_path, 'r') as f:
        sample_metadata = json.load(f)
    
    # Create lookup map for efficiency
    token_to_scene = {s['token']: s['scene_token'] for s in sample_metadata}
    
    # Extract scene_token for every sample in the training set X
    # Each item in X is a dictionary containing 'sample_token'
    try:
        groups = np.array([token_to_scene[sample['sample_token']] for sample in X])
    except KeyError as e:
        raise KeyError(f"Sample token found in training data but missing from sample.json: {e}")
    
    # Explicitly clear metadata to free memory
    del sample_metadata
    
    # Initialize the core splitter with Scene-based GroupKFold logic
    base_splitter = SceneBasedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Return wrapped splitter that injects groups automatically
    return SplitterWrapper(base_splitter, groups)
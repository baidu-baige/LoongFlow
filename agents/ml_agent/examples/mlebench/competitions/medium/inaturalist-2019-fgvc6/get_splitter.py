import json
import os
import numpy as np
import pandas as pd
from typing import Any
from sklearn.model_selection import PredefinedSplit, ShuffleSplit

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/inaturalist-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/57a5d42d-daf8-4241-8197-424dd36c00a6/1/executor/output"

# Task-adaptive concrete type definitions
X = pd.DataFrame  # Contains 'path' and taxonomy metadata
y = pd.Series     # Contains 'category_id'

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.
    
    This implementation uses the official iNaturalist 2019 validation set 
    (from val2019.json) to define the split. If the official validation 
    samples are not present in the input data (e.g., in validation_mode), 
    it falls back to a random shuffle split to ensure pipeline continuity.
    """
    
    # Path to the official validation JSON to identify validation samples
    val_json_path = os.path.join(BASE_DATA_PATH, 'val2019.json')
    
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
        
    # Reconstruct the absolute paths for validation images as they are formatted in load_data.
    # The 'file_name' in the JSON (e.g., "train_val2019/Plantae/Magnoliopsida/123.jpg")
    # is transformed in the pipeline to: BASE_DATA_PATH + "train_val_extracted" + "Plantae/Magnoliopsida/123.jpg"
    val_paths = {
        os.path.join(BASE_DATA_PATH, 'train_val_extracted', img['file_name'].split("/", 1)[1])
        for img in val_data['images']
    }
    
    # Identify which rows in the provided training data pool belong to the official validation set
    is_val = X['path'].isin(val_paths)
    
    if is_val.any():
        # Case: Full dataset or significant portion containing official validation samples.
        # PredefinedSplit requires a test_fold array where -1 is training and 0 is validation.
        test_fold = np.full(len(X), -1)
        test_fold[is_val] = 0
        splitter = PredefinedSplit(test_fold)
    else:
        # Case: validation_mode=True (where load_data takes head(200) of training images).
        # Since the first 200 images are from train2019.json, we must force a split 
        # to have a validation set for the downstream training stage.
        # We use ShuffleSplit instead of StratifiedShuffleSplit because with 200 samples 
        # and 1010 classes, stratification is mathematically impossible (most classes have 1 sample).
        splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
    return splitter
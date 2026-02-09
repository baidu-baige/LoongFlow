import json
import os
import pandas as pd
from typing import Tuple

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/inaturalist-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/57a5d42d-daf8-4241-8197-424dd36c00a6/1/executor/output"

# Concrete type definitions for this task
X = pd.DataFrame  # Contains 'path' and taxonomy metadata
y = pd.Series     # Contains 'category_id'
Ids = pd.Series     # Contains 'id' (image id for test)

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the iNaturalist 2019 datasets.
    """
    
    def load_and_merge(json_file: str, extract_dir: str, is_test: bool = False) -> pd.DataFrame:
        json_path = os.path.join(BASE_DATA_PATH, json_file)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        images_df = pd.DataFrame(data['images'])
        
        if not is_test:
            annotations_df = pd.DataFrame(data['annotations'])
            categories_df = pd.DataFrame(data['categories'])
            
            # Merge images with annotations on image id
            # Note: images_df['id'] is the image identifier
            # annotations_df['image_id'] links to image identifier
            df = images_df.merge(annotations_df, left_on='id', right_on='image_id', suffixes=('', '_ann'))
            
            # Merge with categories on category id
            # annotations_df['category_id'] links to categories_df['id']
            df = df.merge(categories_df, left_on='category_id', right_on='id', suffixes=('', '_cat'))
        else:
            df = images_df
            
        # Path resolution: Strip prefix 'train_val2019/' or 'test2019/' and join with extract_dir
        # file_name example: "train_val2019/Plantae/Magnoliopsida/123456.jpg"
        df['path'] = df['file_name'].apply(lambda x: os.path.join(BASE_DATA_PATH, extract_dir, x.split("/", 1)[1]))
        
        # Verify existence of images
        # We do this for all rows to ensure the downstream pipeline has valid data
        mask = df['path'].apply(os.path.exists)
        df = df[mask].reset_index(drop=True)
        
        return df

    # Step 1: Load and prepare training/validation data
    # We combine train and val JSONs into a single pool for the pipeline's loader
    train_df = load_and_merge('train2019.json', 'train_val_extracted', is_test=False)
    val_df = load_and_merge('val2019.json', 'train_val_extracted', is_test=False)
    
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Step 2: Load and prepare test data
    full_test_df = load_and_merge('test2019.json', 'test_extracted', is_test=True)
    
    # Step 3: Apply validation_mode subsetting
    if validation_mode:
        # Sample up to 200 rows for train and test
        # Use head for deterministic behavior in validation mode
        full_train_df = full_train_df.head(200)
        full_test_df = full_test_df.head(200)
        
    # Step 4: Structure data into required return format
    # train: Path and taxonomy
    taxonomy_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    X_train = full_train_df[['path'] + taxonomy_cols]
    y_train = full_train_df['category_id']
    
    # X_test: Path
    X_test = full_test_df[['path']]
    # test_ids: Image ID (needed for submission)
    test_ids = full_test_df['id']
    
    # Ensure targets and indices are standard types
    y_train = y_train.astype(int)
    test_ids = test_ids.astype(int)

    return X_train, y_train, X_test, test_ids
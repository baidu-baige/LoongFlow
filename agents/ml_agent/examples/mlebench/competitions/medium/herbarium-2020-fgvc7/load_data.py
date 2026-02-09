import os
import json
import pandas as pd
from typing import Tuple, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/herbarium-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/cf84c6d3-8647-45ca-b9ff-02e7ed67cf5b/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing file paths and metadata
y = pd.Series         # Main target vector (category_id)
Ids = pd.Series       # Test image IDs

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Herbarium 2020 FGVC7 datasets.
    Extracts Species, Genus, and Family IDs for training images from nested COCO-format metadata.
    """
    train_json_path = os.path.join(BASE_DATA_PATH, "nybg2020/train/metadata.json")
    test_json_path = os.path.join(BASE_DATA_PATH, "nybg2020/test/metadata.json")

    # Step 1: Load and parse training metadata
    print(f"Loading training metadata from {train_json_path}...")
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    # Convert components to DataFrames
    df_train_ann = pd.DataFrame(train_data['annotations'])
    df_train_img = pd.DataFrame(train_data['images'])
    df_train_cat = pd.DataFrame(train_data['categories'])
    
    # Cleanup raw data to save memory
    del train_data

    print("Structuring training taxonomy and merging metadata...")
    # Merge hierarchy: annotations (category_id) -> images (file_name) -> categories (genus, family)
    df_train = pd.merge(df_train_ann, df_train_img, left_on='image_id', right_on='id')
    df_train = pd.merge(df_train, df_train_cat, left_on='category_id', right_on='id', suffixes=('', '_cat'))

    # Encode taxonomic strings to integer IDs for multi-task learning preparation
    print("Encoding genus and family levels...")
    df_train['genus_id'] = pd.factorize(df_train['genus'])[0]
    df_train['family_id'] = pd.factorize(df_train['family'])[0]

    # Helper function to construct and verify paths safely
    def get_full_path(fname, base_dir):
        # Some file_names in metadata might already include .jpg, others might not
        if not fname.lower().endswith('.jpg'):
            fname = fname + '.jpg'
        return os.path.join(BASE_DATA_PATH, base_dir, fname)

    # Construct full file paths
    df_train['file_name_full'] = df_train['file_name'].apply(lambda x: get_full_path(x, "nybg2020/train"))

    # Step 2: Load and parse test metadata
    print(f"Loading test metadata from {test_json_path}...")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    df_test = pd.DataFrame(test_data['images'])
    del test_data

    df_test['file_name_full'] = df_test['file_name'].apply(lambda x: get_full_path(x, "nybg2020/test"))

    # Step 3: Verify path alignment by sampling
    print("Verifying data paths...")
    sample_train = df_train['file_name_full'].iloc[0]
    sample_test = df_test['file_name_full'].iloc[0]
    
    if not os.path.exists(sample_train):
        # Fallback check: if the path contains redundant 'images/' prefix or is missing it
        # Based on the error, the path attempted was .../images/...jpg.jpg
        # Let's check the first sample specifically to handle any nested structure issues
        raise FileNotFoundError(f"Training image path alignment failed: {sample_train} does not exist.")
    if not os.path.exists(sample_test):
        raise FileNotFoundError(f"Test image path alignment failed: {sample_test} does not exist.")

    # Step 4: Apply validation_mode subsetting
    if validation_mode:
        print("Validation mode: subsetting datasets to 200 samples.")
        df_train = df_train.sample(n=min(200, len(df_train)), random_state=42).reset_index(drop=True)
        df_test = df_test.sample(n=min(200, len(df_test)), random_state=42).reset_index(drop=True)

    # Step 5: Structure return values
    X_train = df_train[['file_name_full', 'category_id', 'genus_id', 'family_id']].rename(
        columns={'file_name_full': 'file_name'}
    )
    y_train = df_train['category_id']
    
    X_test = df_test[['file_name_full']].rename(columns={'file_name_full': 'file_name'})
    test_ids = df_test['id']

    print(f"Successfully loaded {len(X_train)} training and {len(X_test)} test samples.")
    return X_train, y_train, X_test, test_ids
import os
import ujson
import pandas as pd
from typing import Tuple, Any

# Base paths provided in the task context
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/herbarium-2022-fgvc9/prepared/public"
OUTPUT_DATA_PATH = "output/5bfa936c-0be4-4e88-95de-92261403881f/1/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads and prepares the Herbarium 2022: Flora of North America dataset.
    
    This function parses the COCO-style JSON metadata, joins the hierarchical taxonomic tables,
    and maps class identifiers into contiguous ranges for multi-task classification heads.
    
    Args:
        validation_mode: If True, returns a small representative subset (200 rows) for testing.
        
    Returns:
        X_train: DataFrame containing image file paths for training.
        y_train: DataFrame containing target labels (category_idx, genus_idx, family_idx) and original category_id.
        X_test: DataFrame containing image file paths for testing.
        test_ids: Series containing the original image identifiers for the test set.
    """
    train_json_path = os.path.join(BASE_DATA_PATH, "train_metadata.json")
    test_json_path = os.path.join(BASE_DATA_PATH, "test_metadata.json")

    # Step 1: Load JSON metadata efficiently using ujson
    print(f"Loading train metadata from {train_json_path}...")
    with open(train_json_path, 'r') as f:
        train_data = ujson.load(f)

    # Step 2: Structure Train Data
    # COCO format: images, annotations, categories, genera
    print("Structuring training dataframes and joining tables...")
    df_ann = pd.DataFrame(train_data['annotations'])
    df_img = pd.DataFrame(train_data['images'])
    df_cat = pd.DataFrame(train_data['categories'])
    
    # Join annotations (labels) with images (file names) and categories (taxonomy)
    # df_ann: [image_id, category_id, genus_id, institution_id]
    # df_img: [image_id, file_name]
    # df_cat: [category_id, family, genus, species, ...]
    df_train = df_ann.merge(df_img, on='image_id').merge(df_cat, on='category_id')
    
    # Step 3: Load and Structure Test Data
    # Handle the case where test_metadata might be a direct list of image objects
    print(f"Loading test metadata from {test_json_path}...")
    with open(test_json_path, 'r') as f:
        test_data = ujson.load(f)
    
    if isinstance(test_data, list):
        print("Test metadata detected as list format.")
        df_test = pd.DataFrame(test_data)
    elif isinstance(test_data, dict):
        print("Test metadata detected as dictionary format.")
        if 'images' in test_data:
            df_test = pd.DataFrame(test_data['images'])
        else:
            # Fallback if the dictionary contains the entries directly
            df_test = pd.DataFrame(test_data)
    else:
        raise TypeError(f"Unexpected JSON format for test_metadata: {type(test_data)}")

    # Ensure test dataframe has necessary columns
    df_test = df_test[['image_id', 'file_name']]

    # Step 4: Create Contiguous ID Mappings (Required for Multi-Task Classification)
    print("Mapping taxonomic labels to contiguous integer ranges...")
    
    # Category mapping [0, 15500]
    unique_cats = sorted(df_train['category_id'].unique())
    cat_map = {cid: i for i, cid in enumerate(unique_cats)}
    df_train['category_idx'] = df_train['category_id'].map(cat_map).astype(int)
    
    # Genus mapping [0, 2563]
    unique_genera = sorted(df_train['genus_id'].unique())
    genus_map = {gid: i for i, gid in enumerate(unique_genera)}
    df_train['genus_idx'] = df_train['genus_id'].map(genus_map).astype(int)
    
    # Family mapping [0, 271]
    unique_families = sorted(df_train['family'].unique())
    fam_map = {fname: i for i, fname in enumerate(unique_families)}
    df_train['family_idx'] = df_train['family'].map(fam_map).astype(int)

    # Step 5: Construct Absolute File Paths
    # The image roots are provided in the directory structure.
    # file_name in JSON is usually relative to the root category folder.
    df_train['file_path'] = df_train['file_name'].apply(lambda x: os.path.join(BASE_DATA_PATH, 'train_images', x))
    df_test['file_path'] = df_test['file_name'].apply(lambda x: os.path.join(BASE_DATA_PATH, 'test_images', x))

    # Path check: verify first file exists to ensure path construction logic is correct
    if not os.path.exists(df_train['file_path'].iloc[0]):
        # Check if files are nested one level deeper (common in some FGVC extracts)
        test_path = os.path.join(BASE_DATA_PATH, 'train_images', 'images', df_train['file_name'].iloc[0])
        if os.path.exists(test_path):
            print("Detected nested directory structure. Adjusting paths.")
            df_train['file_path'] = df_train['file_name'].apply(lambda x: os.path.join(BASE_DATA_PATH, 'train_images', 'images', x))
            df_test['file_path'] = df_test['file_name'].apply(lambda x: os.path.join(BASE_DATA_PATH, 'test_images', 'images', x))

    # Step 6: Prepare Final Return Objects
    X_train = df_train[['file_path']]
    # y_train contains hierarchical targets and the original category_id for downstream evaluation/submission
    y_train = df_train[['category_idx', 'genus_idx', 'family_idx', 'category_id']]
    
    X_test = df_test[['file_path']]
    test_ids = df_test['image_id']

    # Step 7: Apply Validation Logic
    if validation_mode:
        print("Validation mode enabled: sampling 200 rows.")
        # Stratified sampling for train to ensure multiple categories are seen in small subset
        # If stratified fails due to too few samples per class, fall back to random
        try:
            X_train = X_train.groupby(y_train['category_idx'], group_keys=False).apply(lambda x: x.sample(n=1, random_state=42))
            if len(X_train) > 200:
                X_train = X_train.sample(n=200, random_state=42)
        except:
            X_train = X_train.sample(n=min(200, len(X_train)), random_state=42)
            
        y_train = y_train.loc[X_train.index]
        X_test = X_test.sample(n=min(200, len(X_test)), random_state=42)
        test_ids = test_ids.loc[X_test.index]

    print(f"Loading complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, y_train, X_test, test_ids
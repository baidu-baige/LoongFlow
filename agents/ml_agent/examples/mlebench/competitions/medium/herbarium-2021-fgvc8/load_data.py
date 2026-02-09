import os
import json
import pandas as pd
from typing import Tuple, Any
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.DataFrame
Ids = pd.Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/herbarium-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/ec5e2488-9859-4456-b67a-20c4a0b3bb67/1/executor/output"

def _read_json_file(path: str) -> dict:
    """Helper for parallel JSON parsing."""
    print(f"Reading metadata from {path}...")
    with open(path, 'r') as f:
        return json.load(f)

def _verify_and_fix_paths(df: pd.DataFrame, base_subdir: str) -> pd.DataFrame:
    """Verifies image paths and applies logic-based construction if needed."""
    # Check first 5 samples for existence
    verification_count = min(5, len(df))
    valid = True
    for i in range(verification_count):
        if not os.path.exists(df.iloc[i]['path']):
            valid = False
            break
    
    if not valid:
        print(f"Initial path verification failed for {base_subdir}. Applying logic-based construction...")
        if base_subdir == 'train':
            # train/images/<subfolder1>/<subfolder2>/<image id>.jpg
            # sub1+sub2 = category_id (5 digits)
            df['path'] = df.apply(
                lambda r: os.path.join(
                    BASE_DATA_PATH, 'train', 'images',
                    f"{int(r['category_id']) // 100:03d}",
                    f"{int(r['category_id']) % 100:02d}",
                    f"{int(r['image_id'])}.jpg"
                ), axis=1
            )
        else:
            # test/images/<subfolder>/<image id>.jpg
            # sub = image_id // 1000 (3 digits)
            df['path'] = df.apply(
                lambda r: os.path.join(
                    BASE_DATA_PATH, 'test', 'images',
                    f"{int(r['image_id']) // 1000:03d}",
                    f"{int(r['image_id'])}.jpg"
                ), axis=1
            )
        
        # Final verification
        if not os.path.exists(df.iloc[0]['path']):
            raise FileNotFoundError(f"Critical Error: Failed to locate images for {base_subdir} at {df.iloc[0]['path']}")
    
    return df

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Herbarium 2021 dataset metadata and paths.
    """
    prepared_dir = os.path.join(OUTPUT_DATA_PATH, "prepared_metadata")
    os.makedirs(prepared_dir, exist_ok=True)
    cache_train = os.path.join(prepared_dir, "train_metadata.pkl")
    cache_test = os.path.join(prepared_dir, "test_metadata.pkl")

    if os.path.exists(cache_train) and os.path.exists(cache_test):
        print("Loading processed metadata from cache...")
        train_df = pd.read_pickle(cache_train)
        test_df = pd.read_pickle(cache_test)
    else:
        train_meta_path = os.path.join(BASE_DATA_PATH, "train", "metadata.json")
        test_meta_path = os.path.join(BASE_DATA_PATH, "test", "metadata.json")

        # Parallel JSON parsing for I/O efficiency
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                'train': executor.submit(_read_json_file, train_meta_path),
                'test': executor.submit(_read_json_file, test_meta_path)
            }
            train_data = futures['train'].result()
            test_data = futures['test'].result()

        print("Processing training metadata and taxonomic labels...")
        # Category/Taxonomy Mapping
        cat_df = pd.DataFrame(train_data['categories'])
        fam_le = LabelEncoder()
        ord_le = LabelEncoder()
        cat_df['family_id'] = fam_le.fit_transform(cat_df['family'].astype(str))
        cat_df['order_id'] = ord_le.fit_transform(cat_df['order'].astype(str))
        cat_df = cat_df.rename(columns={'id': 'category_id'})[['category_id', 'family_id', 'order_id']]

        # Annotations (Image -> Category)
        ann_df = pd.DataFrame(train_data['annotations'])[['image_id', 'category_id']]

        # Images (Image -> Filename)
        img_df = pd.DataFrame(train_data['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
        img_df['path'] = img_df['file_name'].apply(lambda x: os.path.join(BASE_DATA_PATH, "train", x))

        # Unified training dataframe
        train_df = ann_df.merge(img_df, on='image_id').merge(cat_df, on='category_id')
        train_df = _verify_and_fix_paths(train_df, 'train')

        print("Processing test metadata...")
        test_df = pd.DataFrame(test_data['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
        test_df['path'] = test_df['file_name'].apply(lambda x: os.path.join(BASE_DATA_PATH, "test", x))
        test_df = _verify_and_fix_paths(test_df, 'test')

        # Optimize memory usage
        for col in ['image_id', 'category_id', 'family_id', 'order_id']:
            if col in train_df.columns:
                train_df[col] = train_df[col].astype('int32')
            if col in test_df.columns:
                test_df[col] = test_df[col].astype('int32')

        print(f"Caching full prepared data to {prepared_dir}...")
        train_df.to_pickle(cache_train)
        test_df.to_pickle(cache_test)

    # Apply validation subsetting if requested
    if validation_mode:
        print("Validation mode: subsetting to 200 rows.")
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=42).reset_index(drop=True)

    # Structure data for return
    X_train = train_df[['image_id', 'path']].reset_index(drop=True)
    y_train = train_df[['category_id', 'family_id', 'order_id']].reset_index(drop=True)
    X_test = test_df[['image_id', 'path']].reset_index(drop=True)
    test_ids = test_df['image_id'].reset_index(drop=True)

    print(f"Successfully loaded {len(X_train)} training and {len(X_test)} test records.")
    return X_train, y_train, X_test, test_ids
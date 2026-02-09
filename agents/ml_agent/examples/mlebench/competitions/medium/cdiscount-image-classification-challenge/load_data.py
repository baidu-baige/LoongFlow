import os
import struct
import pandas as pd
import numpy as np
import bson
from multiprocessing import Pool
from typing import Tuple, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/cdiscount-image-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/96ece161-83e4-4d99-a688-ea7a2b1aa242/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature metadata (offsets, lengths, product IDs)
y = pd.DataFrame      # Target indices (category, level1, level2)
Ids = pd.Series       # Product IDs for test alignment

def _parse_bson_chunk(args):
    """Worker function to parse a chunk of BSON documents."""
    path, chunk_offsets = args
    results = []
    with open(path, 'rb') as f:
        for off, size in chunk_offsets:
            f.seek(off)
            doc_bytes = f.read(size)
            try:
                # Standard pymongo bson decode
                doc = bson.BSON(doc_bytes).decode()
            except:
                # Fallback for standalone bson package
                import bson as bson_alt
                doc = bson_alt.loads(doc_bytes)
            
            _id = doc['_id']
            cat_id = doc.get('category_id') # None for test
            num_imgs = len(doc['imgs'])
            results.append((_id, cat_id, off, size, num_imgs))
    return results

def build_index(bson_path: str, is_train: bool, cat_lookup: pd.DataFrame = None) -> pd.DataFrame:
    """Builds an image-level index for the BSON file."""
    print(f"Indexing {bson_path}...")
    
    # Step 1: Sequential scan for document offsets and sizes
    offsets = []
    with open(bson_path, 'rb') as f:
        pos = 0
        while True:
            header = f.read(4)
            if not header:
                break
            size = struct.unpack('<i', header)[0]
            offsets.append((pos, size))
            pos += size
            f.seek(pos)
    
    print(f"Found {len(offsets)} products. Parsing metadata with multiprocessing...")
    
    # Step 2: Parallel parsing of documents
    num_workers = 32
    chunk_size = (len(offsets) + num_workers - 1) // num_workers
    chunks = [offsets[i:i + chunk_size] for i in range(0, len(offsets), chunk_size)]
    
    with Pool(num_workers) as pool:
        chunk_results = pool.map(_parse_bson_chunk, [(bson_path, chunk) for chunk in chunks])
    
    # Flatten results
    product_results = [res for chunk in chunk_results for res in chunk]
    res_df = pd.DataFrame(product_results, columns=['_id', 'category_id', 'offset', 'length', 'num_imgs'])
    
    print("Expanding to image-level index...")
    # Step 3: Expand product-level metadata to image-level
    n_imgs_arr = res_df['num_imgs'].values
    pids = np.repeat(res_df['_id'].values, n_imgs_arr)
    offsets_img = np.repeat(res_df['offset'].values, n_imgs_arr)
    lengths_img = np.repeat(res_df['length'].values, n_imgs_arr)
    img_indices = np.concatenate([np.arange(n) for n in n_imgs_arr])
    
    index_df = pd.DataFrame({
        '_id': pids,
        'img_idx': img_indices.astype(np.int8),
        'offset': offsets_img,
        'length': lengths_img.astype(np.int32)
    })
    
    if is_train and cat_lookup is not None:
        print("Mapping categories...")
        cat_ids = np.repeat(res_df['category_id'].values, n_imgs_arr)
        # Use series map for efficiency
        index_df['cat_idx'] = pd.Series(cat_ids).map(cat_lookup['cat_idx']).values.astype(np.int16)
        index_df['l1_idx'] = pd.Series(cat_ids).map(cat_lookup['l1_idx']).values.astype(np.int8)
        index_df['l2_idx'] = pd.Series(cat_ids).map(cat_lookup['l2_idx']).values.astype(np.int16)
        
    return index_df

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets. Pre-indexes BSON files for efficient random access.
    """
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    train_bson = os.path.join(BASE_DATA_PATH, "train.bson")
    test_bson = os.path.join(BASE_DATA_PATH, "test.bson")
    cat_csv = os.path.join(BASE_DATA_PATH, "category_names.csv")
    
    train_idx_path = os.path.join(OUTPUT_DATA_PATH, "train_index_v2.parquet")
    test_idx_path = os.path.join(OUTPUT_DATA_PATH, "test_index_v2.parquet")
    
    # 1. Prepare Category Mapping using correct column names from EDA
    cat_df = pd.read_csv(cat_csv)
    # Correct columns according to EDA: category_level1, category_level2, category_level3
    cat_df = cat_df.sort_values('category_id').reset_index(drop=True)
    cat_df['cat_idx'] = np.arange(len(cat_df))
    
    l1_map = {name: i for i, name in enumerate(sorted(cat_df['category_level1'].unique()))}
    l2_map = {name: i for i, name in enumerate(sorted(cat_df['category_level2'].unique()))}
    cat_df['l1_idx'] = cat_df['category_level1'].map(l1_map)
    cat_df['l2_idx'] = cat_df['category_level2'].map(l2_map)
    
    cat_lookup = cat_df.set_index('category_id')[['cat_idx', 'l1_idx', 'l2_idx']]
    
    # 2. Build or Load Index
    if not os.path.exists(train_idx_path):
        train_index = build_index(train_bson, is_train=True, cat_lookup=cat_lookup)
        train_index.to_parquet(train_idx_path, compression='snappy')
    else:
        print(f"Loading existing train index from {train_idx_path}")
        train_index = pd.read_parquet(train_idx_path)
        
    if not os.path.exists(test_idx_path):
        test_index = build_index(test_bson, is_train=False)
        test_index.to_parquet(test_idx_path, compression='snappy')
    else:
        print(f"Loading existing test index from {test_idx_path}")
        test_index = pd.read_parquet(test_idx_path)
        
    # 3. Structure into X, y format
    y_train = train_index[['cat_idx', 'l1_idx', 'l2_idx']].copy()
    X_train = train_index.drop(columns=['cat_idx', 'l1_idx', 'l2_idx'])
    
    X_test = test_index
    test_ids = test_index['_id']
    
    # 4. Handle validation_mode
    if validation_mode:
        print("Validation mode: subsetting to 200 rows")
        X_train = X_train.iloc[:200].reset_index(drop=True)
        y_train = y_train.iloc[:200].reset_index(drop=True)
        X_test = X_test.iloc[:200].reset_index(drop=True)
        test_ids = test_ids.iloc[:200].reset_index(drop=True)
        
    print(f"Data Loaded. Train images: {len(X_train)}, Test images: {len(X_test)}")
    return X_train, y_train, X_test, test_ids
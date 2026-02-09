import pandas as pd
import numpy as np
import cv2
import zipfile
import os
import io
from typing import Tuple, Any, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/kuzushiji-recognition/prepared/public"
OUTPUT_DATA_PATH = "output/2cf35106-db22-4d8d-a450-9ac60aada454/1/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads and prepares the Kuzushiji dataset for detection and classification.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
        - X_train: DataFrame containing detection metadata (image_id, labels) 
                   with classification metadata and unicode maps in .attrs
        - y_train: Series of label strings for the detection task
        - X_test: DataFrame containing test image IDs
        - test_ids: Series of test image IDs
    """
    print("Starting data loading stage...")

    # 1. Load Unicode Translation and create mapping
    unicode_trans_path = os.path.join(BASE_DATA_PATH, "unicode_translation.csv")
    df_unicode = pd.read_csv(unicode_trans_path)
    
    # 2. Load Classification Metadata (character-level info)
    class_meta_path = os.path.join(BASE_DATA_PATH, "classification_metadata.csv")
    df_classification = pd.read_csv(class_meta_path)
    
    # Ensure coordinates are integers
    for col in ['x', 'y', 'w', 'h']:
        df_classification[col] = df_classification[col].astype(int)
        
    # Map Unicode to integer IDs (0-4112) based on unique values in training
    unique_unicodes = sorted(df_classification['unicode'].unique())
    unicode_map = {u: i for i, u in enumerate(unique_unicodes)}
    df_classification['label_id'] = df_classification['unicode'].map(unicode_map).astype(int)
    
    # Correct crop_path to current environment
    # Original path: /root/workspace/evolux/output/mlebench/kuzushiji-recognition/prepared/public/extracted_crops_128/...
    # New path: BASE_DATA_PATH/extracted_crops_128/...
    def fix_path(p):
        parts = p.split('extracted_crops_128/')
        if len(parts) > 1:
            return os.path.join(BASE_DATA_PATH, "extracted_crops_128", parts[1])
        return p
    df_classification['crop_path'] = df_classification['crop_path'].apply(fix_path)

    # 3. Load Detection Data (image-level)
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    df_detection = pd.read_csv(train_csv_path)
    df_detection['labels'] = df_detection['labels'].fillna("") # Handle images with no labels

    # 4. Load Test Data
    test_csv_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    df_test = pd.read_csv(test_csv_path)
    # Only need image_id for X_test
    X_test_df = df_test[['image_id']].copy()

    # 5. Image Caching logic (Resize to 1024px and store in RAM)
    # We'll cache images in a dictionary attached to X_train.attrs
    image_cache = {}
    
    def cache_images_from_zip(zip_path, image_ids, target_size=1024):
        cached = {}
        if not os.path.exists(zip_path):
            print(f"Warning: Zip file {zip_path} not found.")
            return cached
            
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Filter filenames in zip that match our image_ids
            namelist = z.namelist()
            id_to_file = {os.path.splitext(os.path.basename(f))[0]: f for f in namelist if f.endswith('.jpg')}
            
            def process_img(img_id):
                if img_id not in id_to_file:
                    return img_id, None
                try:
                    with z.open(id_to_file[img_id]) as f:
                        img_data = f.read()
                        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            return img_id, None
                        # Resize maintaining aspect ratio
                        h, w = img.shape[:2]
                        scale = target_size / max(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        return img_id, img_resized
                except Exception as e:
                    print(f"Error processing {img_id}: {e}")
                    return img_id, None

            # Use thread pool to speed up decoding and resizing
            ids_to_process = [iid for iid in image_ids]
            with ThreadPoolExecutor(max_workers=36) as executor:
                results = list(tqdm(executor.map(process_img, ids_to_process), total=len(ids_to_process), desc=f"Caching {os.path.basename(zip_path)}"))
            
            for iid, img in results:
                if img is not None:
                    cached[iid] = img
        return cached

    # 6. Apply validation_mode subsetting
    if validation_mode:
        print("Validation mode enabled: subsetting data to 200 samples.")
        df_detection = df_detection.head(200).reset_index(drop=True)
        X_test_df = X_test_df.head(200).reset_index(drop=True)
        # For classification, only keep crops belonging to the subsetted images
        valid_train_ids = set(df_detection['image_id'])
        df_classification = df_classification[df_classification['image_id'].isin(valid_train_ids)].reset_index(drop=True)

    # Cache images (Always cache for subset in validation mode, or full if memory allows)
    # Given 440GB RAM, we can cache all 3k training images easily (~3-5GB at 1024px)
    train_zip = os.path.join(BASE_DATA_PATH, "train_images.zip")
    image_cache.update(cache_images_from_zip(train_zip, df_detection['image_id'].unique()))
    
    test_zip = os.path.join(BASE_DATA_PATH, "test_images.zip")
    image_cache.update(cache_images_from_zip(test_zip, X_test_df['image_id'].unique()))

    # 7. Final preparation of return objects
    X_train = df_detection.copy()
    y_train = df_detection['labels'].copy()
    X_test = X_test_df.copy()
    test_ids = X_test_df['image_id'].copy()

    # Attach metadata to X_train attributes
    X_train.attrs['df_classification'] = df_classification
    X_train.attrs['unicode_map'] = unicode_map
    X_train.attrs['unicode_translation'] = df_unicode
    X_train.attrs['image_cache'] = image_cache

    print(f"Data loading complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Character crops in classification df: {len(df_classification)}")
    
    return X_train, y_train, X_test, test_ids
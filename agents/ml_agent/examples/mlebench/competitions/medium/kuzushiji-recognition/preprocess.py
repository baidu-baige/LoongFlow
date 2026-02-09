import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/kuzushiji-recognition/prepared/public"
OUTPUT_DATA_PATH = "output/2cf35106-db22-4d8d-a450-9ac60aada454/1/executor/output"

# Task-adaptive type definitions
# X and y are lists of dictionaries, where each element corresponds to one original image.
# This ensures row alignment: len(X_processed) == len(X_input).
X = List[Dict[str, Any]]
y = List[Dict[str, Any]]

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw Kuzushiji data into model-ready format.
    Each output row corresponds to an input image, containing its tiles and crops.
    """
    print("Starting preprocessing stage...")

    # Extract metadata from attributes
    unicode_map = X_train.attrs.get('unicode_map', {})
    df_class_full = X_train.attrs.get('df_classification', pd.DataFrame())
    
    # Parameters from Technical Specification
    TILE_SIZE = 1024
    OVERLAP = 128
    STRIDE = TILE_SIZE - OVERLAP
    CROP_SIZE = 128

    def parse_labels(label_str: str) -> List[Dict]:
        if not label_str or pd.isna(label_str) or label_str == "":
            return []
        parts = label_str.split()
        labels = []
        for i in range(0, len(parts), 5):
            labels.append({
                'unicode': parts[i],
                'x': int(parts[i+1]),
                'y': int(parts[i+2]),
                'w': int(parts[i+3]),
                'h': int(parts[i+4]),
                'label_id': unicode_map.get(parts[i], -1)
            })
        return labels

    def get_tile_offsets(size: int, tile_size: int, stride: int) -> List[int]:
        if size <= tile_size:
            return [0]
        offsets = list(range(0, size - tile_size, stride))
        if offsets[-1] + tile_size < size:
            offsets.append(size - tile_size)
        return sorted(list(set(offsets)))

    def pad_and_resize_crop(crop: np.ndarray, target_size: int) -> np.ndarray:
        h, w = crop.shape[:2]
        if h == w:
            return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            crop = cv2.copyMakeBorder(crop, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    def process_single_image(img_id: str, label_str: str, zip_path: str, char_metadata: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Processes one image: returns data dictionary and label dictionary."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                target_file = None
                for f in z.namelist():
                    if img_id in f and f.endswith('.jpg'):
                        target_file = f
                        break
                if not target_file:
                    raise FileNotFoundError(f"Image {img_id} not found in {zip_path}")
                
                with z.open(target_file) as f:
                    img_data = f.read()
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError(f"Failed to decode image {img_id}")

            H, W = img.shape[:2]
            parsed_labels = parse_labels(label_str)
            
            # 1. Tiling for Detection
            tiles_x = get_tile_offsets(W, TILE_SIZE, STRIDE)
            tiles_y = get_tile_offsets(H, TILE_SIZE, STRIDE)
            
            img_tiles = []
            img_tile_labels = []
            
            for ty in tiles_y:
                for tx in tiles_x:
                    tile_img = img[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE].copy()
                    if tile_img.shape[0] < TILE_SIZE or tile_img.shape[1] < TILE_SIZE:
                        tile_img = cv2.copyMakeBorder(tile_img, 0, TILE_SIZE-tile_img.shape[0], 
                                                     0, TILE_SIZE-tile_img.shape[1], 
                                                     cv2.BORDER_CONSTANT, value=0)
                    
                    tile_labels = []
                    for lbl in parsed_labels:
                        cx, cy = lbl['x'] + lbl['w']//2, lbl['y'] + lbl['h']//2
                        if tx <= cx < tx + TILE_SIZE and ty <= cy < ty + TILE_SIZE:
                            tile_labels.append({
                                'label_id': lbl['label_id'],
                                'rx': cx - tx,
                                'ry': cy - ty,
                                'rw': lbl['w'],
                                'rh': lbl['h']
                            })
                    
                    img_tiles.append(tile_img)
                    img_tile_labels.append(tile_labels)

            # 2. Cropping for Classification
            img_crops = []
            img_crop_labels = []
            for _, row in char_metadata.iterrows():
                x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
                x_start, y_start = max(0, x), max(0, y)
                x_end, y_end = min(W, x+w), min(H, y+h)
                if x_end > x_start and y_end > y_start:
                    crop = img[y_start:y_end, x_start:x_end]
                    img_crops.append(pad_and_resize_crop(crop, CROP_SIZE))
                    img_crop_labels.append(row['label_id'])

            x_out = {
                'image_id': img_id,
                'tiles': img_tiles,
                'crops': img_crops,
                'orig_shape': (H, W)
            }
            y_out = {
                'tile_labels': img_tile_labels,
                'crop_labels': img_crop_labels
            }
            return x_out, y_out
        except Exception as e:
            print(f"Critical error in process_single_image for {img_id}: {e}")
            # Return empty but aligned structures to satisfy row requirement if error occurs
            return {'image_id': img_id, 'tiles': [], 'crops': [], 'orig_shape': (0,0)}, {'tile_labels': [], 'crop_labels': []}

    def process_set(df: pd.DataFrame, labels_ser: pd.Series, zip_path: str) -> Tuple[X, y]:
        image_ids = df['image_id'].tolist()
        label_strings = [labels_ser.iloc[i] if i < len(labels_ser) else "" for i in range(len(df))]
        
        set_ids = set(image_ids)
        set_char_meta = df_class_full[df_class_full['image_id'].isin(set_ids)]
        
        X_list = [None] * len(image_ids)
        y_list = [None] * len(image_ids)
        
        print(f"Processing {len(image_ids)} images from {os.path.basename(zip_path)}...")
        
        with ThreadPoolExecutor(max_workers=36) as executor:
            futures = []
            for i, (iid, lstr) in enumerate(zip(image_ids, label_strings)):
                char_meta = set_char_meta[set_char_meta['image_id'] == iid]
                futures.append(executor.submit(process_single_image, iid, lstr, zip_path, char_meta))
            
            for i, future in enumerate(futures):
                x_res, y_res = future.result()
                X_list[i] = x_res
                y_list[i] = y_res

        return X_list, y_list

    # Process all splits
    train_zip = os.path.join(BASE_DATA_PATH, "train_images.zip")
    test_zip = os.path.join(BASE_DATA_PATH, "test_images.zip")

    X_train_proc, y_train_proc = process_set(X_train, y_train, train_zip)
    X_val_proc, y_val_proc = process_set(X_val, y_val, train_zip)
    # y_test is not used, but we return dummy to keep format
    X_test_proc, y_test_dummy = process_set(X_test, pd.Series([""]*len(X_test)), test_zip)

    print(f"Preprocessing complete. Row alignment: Train({len(X_train_proc)}=={len(y_train_proc)}), Val({len(X_val_proc)}=={len(y_val_proc)}), Test({len(X_test_proc)})")
    
    return X_train_proc, y_train_proc, X_val_proc, y_val_proc, X_test_proc
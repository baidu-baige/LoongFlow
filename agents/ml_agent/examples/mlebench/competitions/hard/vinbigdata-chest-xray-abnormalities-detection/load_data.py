import os
import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Any, List
from ensemble_boxes import weighted_boxes_fusion

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-10/evolux/output/mlebench/vinbigdata-chest-xray-abnormalities-detection/prepared/public"
OUTPUT_DATA_PATH = "output/7357c439-e7eb-4cf1-afb0-36b77c92c672/1/executor/output"

def _process_single_dicom(item):
    """
    Worker function to convert a single DICOM file to PNG with VOI LUT application.
    """
    dicom_path, png_path = item
    if os.path.exists(png_path):
        return
    try:
        ds = pydicom.dcmread(dicom_path)
        # Apply VOI LUT (Value of Interest Look-Up Table) for better contrast
        img = apply_voi_lut(ds.pixel_array, ds)
        
        # Handle Photometric Interpretation
        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = np.amax(img) - img
            
        # Standardize and scale to 8-bit
        img = img.astype(np.float32)
        img = img - np.min(img)
        if np.max(img) > 0:
            img = (img / np.max(img) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        cv2.imwrite(png_path, img)
    except Exception as e:
        # Propagate error immediately to signal processing failure
        raise RuntimeError(f"Critical error processing DICOM {dicom_path}: {e}")

def load_data(validation_mode: bool = False) -> Tuple[pd.DataFrame, List[np.ndarray], pd.DataFrame, pd.Series]:
    """
    Loads and prepares the VinBigData Chest X-ray dataset.
    
    Returns:
        X_train (pd.DataFrame): Metadata and paths for training images.
        y_train (List[np.ndarray]): Consolidated bounding boxes [x_min, y_min, x_max, y_max, class_id].
        X_test (pd.DataFrame): Metadata and paths for test images.
        test_ids (pd.Series): Image identifiers for test set.
    """
    print("Stage 1: Initializing data loading and preparation pipeline...")
    
    # 1. Path Definitions
    train_dicom_dir = os.path.join(BASE_DATA_PATH, "train")
    test_dicom_dir = os.path.join(BASE_DATA_PATH, "test")
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    meta_csv_path = os.path.join(BASE_DATA_PATH, "image_metadata_full.csv")
    
    prep_dir = os.path.join(OUTPUT_DATA_PATH, "prepared_data")
    train_png_dir = os.path.join(prep_dir, "train_png")
    test_png_dir = os.path.join(prep_dir, "test_png")
    consensus_csv_path = os.path.join(prep_dir, "train_consensus.csv")
    
    os.makedirs(train_png_dir, exist_ok=True)
    os.makedirs(test_png_dir, exist_ok=True)

    # 2. DICOM to PNG Conversion (Full Preparation)
    # This stage is executed regardless of validation_mode.
    train_files = [f for f in os.listdir(train_dicom_dir) if f.endswith(".dicom")]
    test_files = [f for f in os.listdir(test_dicom_dir) if f.endswith(".dicom")]
    
    processing_tasks = []
    for f in train_files:
        processing_tasks.append((os.path.join(train_dicom_dir, f), 
                                 os.path.join(train_png_dir, f.replace(".dicom", ".png"))))
    for f in test_files:
        processing_tasks.append((os.path.join(test_dicom_dir, f), 
                                 os.path.join(test_png_dir, f.replace(".dicom", ".png"))))
    
    if processing_tasks:
        print(f"Executing conversion of {len(processing_tasks)} DICOM files using 36 parallel workers...")
        with ProcessPoolExecutor(max_workers=36) as executor:
            # Force execution to ensure all files are processed and any errors propagate
            list(executor.map(_process_single_dicom, processing_tasks))
        print("DICOM to PNG conversion completed successfully.")

    # 3. Annotation Consolidation (Weighted Boxes Fusion)
    if not os.path.exists(consensus_csv_path):
        print("Consolidating multi-radiologist annotations using Weighted Boxes Fusion (WBF)...")
        train_df = pd.read_csv(train_csv_path)
        meta_df = pd.read_csv(meta_csv_path)
        train_df = train_df.merge(meta_df[['image_id', 'width', 'height']], on='image_id', how='left')
        
        results = []
        image_ids = train_df['image_id'].unique()
        
        for img_id in image_ids:
            img_group = train_df[train_df['image_id'] == img_id]
            w, h = img_group['width'].iloc[0], img_group['height'].iloc[0]
            
            findings = img_group[img_group['class_id'] != 14]
            if len(findings) == 0:
                # Image has only 'No finding' annotations
                results.append({'image_id': img_id, 'class_id': 14, 
                                'x_min': 0.0, 'y_min': 0.0, 'x_max': 1.0, 'y_max': 1.0})
                continue
            
            # Organize boxes per radiologist for WBF
            boxes_list, scores_list, labels_list = [], [], []
            for rad_id in findings['rad_id'].unique():
                rad_df = findings[findings['rad_id'] == rad_id]
                b = rad_df[['x_min', 'y_min', 'x_max', 'y_max']].values.copy()
                # Normalize coordinates to [0, 1] for WBF
                b[:, [0, 2]] /= w
                b[:, [1, 3]] /= h
                boxes_list.append(np.clip(b, 0, 1).tolist())
                scores_list.append([1.0] * len(rad_df))
                labels_list.append(rad_df['class_id'].values.tolist())
            
            # Apply WBF consolidation (iou_thr=0.5 as specified)
            merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.0001
            )
            
            # Re-scale to absolute pixels
            merged_boxes[:, [0, 2]] *= w
            merged_boxes[:, [1, 3]] *= h
            
            for i in range(len(merged_boxes)):
                results.append({
                    'image_id': img_id,
                    'class_id': int(merged_labels[i]),
                    'x_min': merged_boxes[i][0],
                    'y_min': merged_boxes[i][1],
                    'x_max': merged_boxes[i][2],
                    'y_max': merged_boxes[i][3]
                })
        
        consensus_df = pd.DataFrame(results)
        consensus_df.to_csv(consensus_csv_path, index=False)
        print(f"Consolidated labels saved to {consensus_csv_path}")
    else:
        print(f"Loading existing consensus file: {consensus_csv_path}")
        consensus_df = pd.read_csv(consensus_csv_path)

    # 4. Final Dataset Assembly
    meta_df = pd.read_csv(meta_csv_path)
    
    # Train assembly
    available_train_ids = [f.replace(".png", "") for f in os.listdir(train_png_dir)]
    X_train = meta_df[meta_df['image_id'].isin(available_train_ids)].copy()
    X_train['filepath'] = X_train['image_id'].apply(lambda x: os.path.join(train_png_dir, f"{x}.png"))
    X_train = X_train.sort_values('image_id').reset_index(drop=True)
    
    # Align y_train with X_train rows
    y_train = []
    for img_id in X_train['image_id']:
        boxes = consensus_df[consensus_df['image_id'] == img_id][['x_min', 'y_min', 'x_max', 'y_max', 'class_id']].values
        y_train.append(boxes)
    
    # Test assembly
    available_test_ids = [f.replace(".png", "") for f in os.listdir(test_png_dir)]
    X_test = meta_df[meta_df['image_id'].isin(available_test_ids)].copy()
    X_test['filepath'] = X_test['image_id'].apply(lambda x: os.path.join(test_png_dir, f"{x}.png"))
    X_test = X_test.sort_values('image_id').reset_index(drop=True)
    test_ids = X_test['image_id']

    # 5. Handle Validation Mode
    if validation_mode:
        print("Validation mode enabled: Returning 200 samples for rapid prototyping.")
        X_train = X_train.head(200)
        y_train = y_train[:200]
        X_test = X_test.head(200)
        test_ids = test_ids.head(200)

    print(f"Data loading completed. Train: {len(X_train)} samples | Test: {len(X_test)} samples.")
    return X_train, y_train, X_test, test_ids
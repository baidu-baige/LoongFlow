import os
import pandas as pd
import numpy as np
import pydicom
import cv2
from typing import Tuple, Any, List
from joblib import Parallel, delayed
from pathlib import Path

# Paths based on provided context
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/rsna-miccai-brain-tumor-radiogenomic-classification/prepared/public"
OUTPUT_DATA_PATH = "output/dafb557f-655e-4395-9835-6f75549a5b27/1/executor/output"

# Concrete types for this task
# X: 5D NumPy array (N_samples, C_modalities, Depth, Height, Width)
# y: 1D NumPy array (N_samples,)
# Ids: 1D NumPy array (N_samples,)
X = np.ndarray
y = np.ndarray
Ids = np.ndarray

def load_dicom_image(path: str, img_size: int = 256) -> np.ndarray:
    """
    Reads a single DICOM file, crops black borders, and resizes to target resolution.
    """
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array
        
        # Min-max normalization for the individual slice to improve contrast
        max_val = np.max(data)
        min_val = np.min(data)
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        else:
            data = np.zeros_like(data, dtype=np.float32)
        
        # Scale to 0-255 for bbox calculation
        data_uint8 = (data * 255).astype(np.uint8)
        
        # Bounding box crop: Find all non-zero pixel coordinates
        coords = np.argwhere(data_uint8 > 0)
        if coords.size > 0:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            data = data[ymin:ymax+1, xmin:xmax+1]
        
        # Resize to standardized square resolution using linear interpolation
        if data.shape[0] > 0 and data.shape[1] > 0:
            return cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        else:
            return np.zeros((img_size, img_size), dtype=np.float32)
    except Exception:
        # Return a zero-filled array if the slice is corrupted/unreadable
        return np.zeros((img_size, img_size), dtype=np.float32)

def load_dicom_3d(patient_id: Any, modality: str, split: str, img_size: int = 256, num_slices: int = 64) -> np.ndarray:
    """
    Constructs a 3D volumetric array for a patient's specific MRI modality.
    """
    patient_dir = str(patient_id).zfill(5)
    modality_path = os.path.join(BASE_DATA_PATH, split, patient_dir, modality)
    
    # Identify and sort DICOM files by the numeric sequence in filenames (e.g., Image-X.dcm)
    def parse_filename(p: Path) -> int:
        try:
            # Filenames are typically 'Image-1.dcm', 'Image-2.dcm', etc.
            return int(p.stem.split("-")[-1])
        except (ValueError, IndexError):
            return 0
            
    files = sorted(Path(modality_path).glob("*.dcm"), key=parse_filename)
    
    if not files:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)
    
    # Uniformly select indices to reach target depth (num_slices)
    indices = np.linspace(0, len(files) - 1, num_slices).astype(int)
    
    volume = []
    for idx in indices:
        img = load_dicom_image(str(files[idx]), img_size)
        volume.append(img)
        
    return np.stack(volume).astype(np.float32)

def process_patient_unified(patient_id: Any, split: str, modalities: List[str], img_size: int, num_slices: int) -> np.ndarray:
    """
    Processes all modalities and stacks them into a (C, D, H, W) array.
    """
    volumes = []
    for mod in modalities:
        vol = load_dicom_3d(patient_id, mod, split, img_size, num_slices)
        volumes.append(vol)
    return np.stack(volumes) # Shape: (4, 64, 256, 256)

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the brain tumor MRI datasets.
    Converts DICOM scans into a unified 5D NumPy array (N, C, D, H, W).
    """
    print(f"Starting data loading. validation_mode={validation_mode}")
    
    # Configuration
    modalities = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    depth, height, width = 64, 256, 256
    excluded_cases = [109, 123, 709]
    
    # Load labels
    train_labels_file = os.path.join(BASE_DATA_PATH, "train_labels.csv")
    sample_sub_file = os.path.join(BASE_DATA_PATH, "sample_submission.csv")
    
    train_df = pd.read_csv(train_labels_file)
    test_df = pd.read_csv(sample_sub_file)
    
    # Filter known problematic cases
    train_df = train_df[~train_df['BraTS21ID'].isin(excluded_cases)].reset_index(drop=True)
    
    # Apply validation subsetting
    if validation_mode:
        train_df = train_df.head(200)
        test_df = test_df.head(200)
        print(f"Limiting to 200 samples for validation mode.")

    # Process Training Data in Parallel
    print(f"Processing {len(train_df)} training samples...")
    train_list = Parallel(n_jobs=36)(
        delayed(process_patient_unified)(pid, 'train', modalities, height, depth)
        for pid in train_df['BraTS21ID']
    )
    X_train = np.stack(train_list)  # Shape: (N, 4, 64, 256, 256)
    y_train = train_df['MGMT_value'].values.astype(np.int64)
    
    # Process Test Data in Parallel
    print(f"Processing {len(test_df)} test samples...")
    test_list = Parallel(n_jobs=36)(
        delayed(process_patient_unified)(pid, 'test', modalities, height, depth)
        for pid in test_df['BraTS21ID']
    )
    X_test = np.stack(test_list)    # Shape: (M, 4, 64, 256, 256)
    test_ids = test_df['BraTS21ID'].values
    
    # Final validation of alignment
    assert len(X_train) == len(y_train), f"Mismatch: X_train {len(X_train)}, y_train {len(y_train)}"
    assert len(X_test) == len(test_ids), f"Mismatch: X_test {len(X_test)}, test_ids {len(test_ids)}"
    
    print(f"Loading complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, y_train, X_test, test_ids
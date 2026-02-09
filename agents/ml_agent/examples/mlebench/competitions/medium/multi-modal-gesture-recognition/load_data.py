import os
import tarfile
import zipfile
import io
import pandas as pd
import numpy as np
import scipy.io
from scipy.io import wavfile
import pickle
from typing import Tuple, List, Dict, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

# Path configuration
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/multi-modal-gesture-recognition/prepared/public"
OUTPUT_DATA_PATH = "output/7d9b4fa5-39b3-4a58-b088-17228f41073d/11/executor/output"
PREPARED_DATA_DIR = os.path.join(OUTPUT_DATA_PATH, "prepared_data")

# Task-adaptive type definitions
# X: List of dicts, each containing 'skeleton', 'audio', 'fs', 'id'
# y: List of lists of integers (the sequence of gesture IDs)
# Ids: List of 4-digit strings for alignment
X = List[Dict[str, Any]]
y = List[List[int]]
Ids = List[str]

GESTURE_MAP = {
    'vattene': 1, 'vieniqui': 2, 'perfetto': 3, 'furbo': 4, 'cheduepalle': 5,
    'chevuoi': 6, 'daccordo': 7, 'seipazzo': 8, 'combinato': 9, 'freganiente': 10,
    'ok': 11, 'cosatifarei': 12, 'basta': 13, 'prendere': 14, 'noncenepiu': 15,
    'fame': 16, 'tantotempo': 17, 'buonissimo': 18, 'messidaccordo': 19, 'sonostufo': 20
}

def parse_mat_file(mat_bytes: bytes) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Parses skeleton data and temporal labels from Matlab bytes."""
    try:
        with io.BytesIO(mat_bytes) as f:
            mat_data = scipy.io.loadmat(f)
        
        # Access the nested 'Video' structure
        video = mat_data['Video'][0, 0]
        frames = video['Frames'][0]
        num_frames = len(frames)
        # Skeleton: (num_frames, 20 joints, 3 coordinates)
        skeleton_seq = np.zeros((num_frames, 20, 3), dtype=np.float32)
        
        for i in range(num_frames):
            try:
                skel_container = frames[i]['Skeleton']
                if skel_container.size > 0:
                    pos = skel_container[0, 0]['WorldPosition']
                    if pos.shape == (20, 3):
                        skeleton_seq[i] = pos
            except:
                continue
        
        precise_labels = []
        if 'Labels' in video.dtype.names:
            labels_obj = video['Labels']
            if labels_obj.size > 0:
                for label_struct in labels_obj[0]:
                    try:
                        name = str(label_struct['Name'][0])
                        precise_labels.append({
                            'name': name,
                            'id': GESTURE_MAP.get(name, -1),
                            'begin': int(label_struct['Begin'][0, 0]),
                            'end': int(label_struct['End'][0, 0])
                        })
                    except:
                        continue
        return skeleton_seq, precise_labels
    except Exception as e:
        return np.zeros((0, 20, 3), dtype=np.float32), []

def process_sample_zip(zip_bytes: bytes, sid: str) -> Dict[str, Any]:
    """Extracts modalities from the Sample ZIP contained within archives."""
    try:
        with io.BytesIO(zip_bytes) as f:
            with zipfile.ZipFile(f) as z:
                names = z.namelist()
                mat_f = next((n for n in names if n.endswith('_data.mat')), None)
                wav_f = next((n for n in names if n.endswith('_audio.wav')), None)
                
                skeleton, plabels = parse_mat_file(z.read(mat_f)) if mat_f else (np.zeros((0, 20, 3)), [])
                fs, audio = 0, np.array([], dtype=np.float32)
                if wav_f:
                    with io.BytesIO(z.read(wav_f)) as wf:
                        fs, audio = wavfile.read(wf)
                        
        return {
            'id': sid,
            'skeleton': skeleton,
            'audio': audio,
            'fs': int(fs),
            'precise_labels': plabels
        }
    except Exception:
        return {'id': sid, 'skeleton': np.zeros((0, 20, 3)), 'audio': np.array([]), 'fs': 0, 'precise_labels': []}

def extract_from_tar(tar_path: str, targets: set) -> Dict[str, Dict[str, Any]]:
    """Scans tar.gz for targeted Sample ZIPs based on IDs."""
    results = {}
    print(f"Opening archive: {os.path.basename(tar_path)}")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".zip"):
                    base_name = os.path.basename(member.name)
                    # Extract 4-digit ID from filename like Sample00001.zip or Sample0001.zip
                    sid_clean = base_name.replace(".zip", "").replace("Sample", "")
                    sid = sid_clean[-4:] if len(sid_clean) >= 4 else sid_clean.zfill(4)
                    
                    if sid in targets:
                        f = tar.extractfile(member)
                        if f:
                            results[sid] = process_sample_zip(f.read(), sid)
    except Exception as e:
        print(f"Error processing archive {tar_path}: {e}")
    return results

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the multi-modal gesture recognition task.
    """
    os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
    cache_file = os.path.join(PREPARED_DATA_DIR, "multimodal_features.pkl")

    # 1. Load Metadata CSVs
    train_csv_path = os.path.join(BASE_DATA_PATH, "training.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")
    
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Metadata CSVs not found in {BASE_DATA_PATH}")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Standardize IDs to 4-character strings
    train_df['Id_str'] = train_df['Id'].astype(str).str.zfill(4)
    test_df['Id_str'] = test_df['Id'].astype(str).str.zfill(4)
    
    target_ids = set(train_df['Id_str']) | set(test_df['Id_str'])

    # 2. Preparation Logic (Always Full)
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        with open(cache_file, 'rb') as f:
            all_samples = pickle.load(f)
    else:
        print("Starting parallel data extraction from raw archives...")
        archives = [os.path.join(BASE_DATA_PATH, f) for f in os.listdir(BASE_DATA_PATH) if f.endswith(".tar.gz")]
        if not archives:
            raise FileNotFoundError("No .tar.gz archives found in the data directory.")

        all_samples = {}
        # Utilizing 32 out of 36 cores for parallel decompressing and parsing
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(extract_from_tar, arch, target_ids) for arch in archives]
            for future in as_completed(futures):
                all_samples.update(future.result())

        print(f"Preparation complete. Processed {len(all_samples)} samples. Caching...")
        with open(cache_file, 'wb') as f:
            pickle.dump(all_samples, f)

    # 3. Align and Structure Data
    X_train: X = []
    y_train: y = []
    X_test: X = []
    test_ids: Ids = []

    # Align Training
    for _, row in train_df.iterrows():
        sid = row['Id_str']
        if sid in all_samples:
            X_train.append(all_samples[sid])
            # Sequence format: "2 14 20" -> [2, 14, 20]
            y_train.append([int(i) for i in str(row['Sequence']).split()])
        else:
            print(f"Warning: Training sample {sid} missing from archives.")

    # Align Test
    for _, row in test_df.iterrows():
        sid = row['Id_str']
        if sid in all_samples:
            X_test.append(all_samples[sid])
            test_ids.append(sid)
        else:
            print(f"Warning: Test sample {sid} missing from archives.")

    # 4. Handle validation_mode (Subsetting)
    if validation_mode:
        print("Validation mode enabled. Returning subset of 200 samples.")
        X_train = X_train[:200]
        y_train = y_train[:200]
        X_test = X_test[:min(200, len(X_test))]
        test_ids = test_ids[:min(200, len(test_ids))]

    # Final integrity check
    if not X_train or not X_test:
        raise RuntimeError("Data loading failed: Training or Test sets are empty.")
    if len(X_train) != len(y_train):
        raise ValueError(f"Train alignment mismatch: X({len(X_train)}) vs y({len(y_train)})")
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test alignment mismatch: X({len(X_test)}) vs ids({len(test_ids)})")

    print(f"Successfully loaded {len(X_train)} training and {len(X_test)} test samples.")
    return X_train, y_train, X_test, test_ids
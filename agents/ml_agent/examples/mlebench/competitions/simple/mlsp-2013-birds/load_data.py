import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

# Global paths based on environment specification
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-10/evolux/output/mlebench/mlsp-2013-birds/prepared/public"
OUTPUT_DATA_PATH = "output/9f7a14b2-9e2e-4beb-a8af-238199431c62/57/executor/output"

# Task-adaptive type definitions
X = List[Dict[str, Any]]  # List of dictionaries containing multi-modal features per recording
y = np.ndarray            # (N, 19) float32 array of multi-labels
Ids = np.ndarray          # (N,) int32 array of recording identifiers (rec_id)

def parse_filename_meta(filename: str) -> Dict[str, Any]:
    """
    Parses Site, Month, and Hour from filename pattern: SITE_YYYYMMDD_HHMMSS
    Example: 'PC1_20090531_050000.wav' -> {'site': 'PC1', 'month': 5, 'hour': 5}
    """
    try:
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        site = parts[0]
        date_part = parts[1]
        time_part = parts[2]
        
        month = int(date_part[4:6])
        hour = int(time_part[0:2])
        return {'site': site, 'month': month, 'hour': hour}
    except (IndexError, ValueError):
        return {'site': 'Unknown', 'month': 0, 'hour': 0}

def robust_read_numeric(path: str) -> pd.DataFrame:
    """
    Reads numeric data files that might have mixed delimiters or headers.
    Ensures robust parsing by handling comma/space delimiters.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file not found: {path}")
        
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().replace(',', ' ')
            if not line:
                continue
            parts = line.split()
            try:
                # Try to convert elements to float, skip line if first element isn't numeric
                row = [float(x) for x in parts]
                rows.append(row)
            except (ValueError, IndexError):
                # Skip header or corrupted lines
                continue
    return pd.DataFrame(rows)

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the bird audio dataset with multi-modal features.
    Harvests: Site, Month, Hour, 100D Histograms, 38D Segment Features, and 3D Geometric traits.
    """
    print(f"Stage 1: Initializing data harvester (validation_mode={validation_mode})...")
    
    essential_dir = os.path.join(BASE_DATA_PATH, "essential_data")
    supp_dir = os.path.join(BASE_DATA_PATH, "supplemental_data")
    
    # Path verification
    fold_path = os.path.join(essential_dir, "CVfolds_2.txt")
    label_path = os.path.join(essential_dir, "rec_labels_test_hidden.txt")
    mapping_path = os.path.join(essential_dir, "rec_id2filename.txt")
    hist_path = os.path.join(supp_dir, "histogram_of_segments.txt")
    seg_feat_path = os.path.join(supp_dir, "segment_features.txt")
    seg_rect_path = os.path.join(supp_dir, "segment_rectangles.txt")
    spec_dir = os.path.join(supp_dir, "spectrograms")

    for p in [fold_path, label_path, mapping_path, hist_path, seg_feat_path, seg_rect_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required resource: {p}")

    # 1. Load CV Folds (0=Train, 1=Test)
    print("Loading split folds...")
    df_fold = robust_read_numeric(fold_path)
    df_fold.columns = ['rec_id', 'fold']
    df_fold = df_fold.astype(int)
    
    # 2. Load Mapping and Parse Metadata
    print("Parsing recording metadata and mapping BMP paths...")
    mapping_dict = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('rec_id'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    rid = int(parts[0])
                    fname = parts[1].strip()
                    mapping_dict[rid] = fname
                except ValueError:
                    continue
                    
    meta_dict = {}
    bmp_path_dict = {}
    for rid, fname in mapping_dict.items():
        meta_dict[rid] = parse_filename_meta(fname)
        bmp_name = os.path.splitext(fname)[0] + ".bmp"
        potential_path = os.path.join(spec_dir, bmp_name)
        if os.path.exists(potential_path):
            bmp_path_dict[rid] = potential_path
        else:
            # Propagate error if primary signal is missing
            raise FileNotFoundError(f"Spectrogram BMP not found for rec_id {rid}: {potential_path}")

    # 3. Load Labels (Multi-label, 19 species)
    print("Parsing multi-labels...")
    label_dict = {}
    num_species = 19
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('rec_id'):
                continue
            parts = line.split(',')
            try:
                rid = int(parts[0])
                if '?' in line:
                    label_dict[rid] = None
                else:
                    target = np.zeros(num_species, dtype=np.float32)
                    for p in parts[1:]:
                        p_clean = p.strip()
                        if p_clean and p_clean.isdigit():
                            idx = int(p_clean)
                            if 0 <= idx < num_species:
                                target[idx] = 1.0
                    label_dict[rid] = target
            except (ValueError, IndexError):
                continue

    # 4. Load 100D Histograms
    print("Loading 100D acoustic histograms...")
    df_hist = robust_read_numeric(hist_path)
    # First column is rec_id, remaining 100 are features
    hist_dict = {int(row[0]): row.iloc[1:101].values.astype(np.float32) for _, row in df_hist.iterrows()}

    # 5. Load 38D Segment Features
    print("Loading 38D segment descriptors...")
    df_seg_feat = robust_read_numeric(seg_feat_path)
    # Format: rec_id, segment_id, feature_1, ..., feature_38
    seg_feat_groups = df_seg_feat.groupby(df_seg_feat.columns[0])
    seg_feat_dict = {int(rid): group.iloc[:, 2:].values.astype(np.float32) for rid, group in seg_feat_groups}

    # 6. Load Segment Rectangles and Calculate 3D Geometric Traits
    print("Calculating 3D geometric traits: duration, bandwidth, area...")
    df_rect = robust_read_numeric(seg_rect_path)
    # Format: rec_id, segment_id, min_x, max_x, min_y, max_y
    df_rect.columns = ['rec_id', 'seg_id', 'min_x', 'max_x', 'min_y', 'max_y']
    
    df_rect['duration'] = df_rect['max_x'] - df_rect['min_x']
    df_rect['bandwidth'] = df_rect['max_y'] - df_rect['min_y']
    df_rect['area'] = df_rect['duration'] * df_rect['bandwidth']
    
    geo_traits_groups = df_rect.groupby('rec_id')
    geo_dict = {int(rid): group[['duration', 'bandwidth', 'area']].values.astype(np.float32) for rid, group in geo_traits_groups}

    # 7. Assemble datasets
    print("Assembling multi-modal telemetry structures...")
    X_train_list: X = []
    y_train_list: List[np.ndarray] = []
    X_test_list: X = []
    test_ids_list: List[int] = []

    # Use CVfolds_2.txt as the master index for recordings
    for _, row in df_fold.sort_values('rec_id').iterrows():
        rid = int(row['rec_id'])
        fold = int(row['fold'])
        
        # Aggregate all harvester signals for this ID
        sample_data = {
            'rec_id': rid,
            'meta': meta_dict.get(rid, {'site': 'Unknown', 'month': 0, 'hour': 0}),
            'hist': hist_dict.get(rid, np.zeros(100, dtype=np.float32)),
            'seg': seg_feat_dict.get(rid, np.zeros((0, 38), dtype=np.float32)),
            'geo': geo_dict.get(rid, np.zeros((0, 3), dtype=np.float32)),
            'bmp_path': bmp_path_dict.get(rid, "")
        }
        
        if fold == 0:  # Training set
            target = label_dict.get(rid)
            if target is not None:
                X_train_list.append(sample_data)
                y_train_list.append(target)
        else:  # Test set
            X_test_list.append(sample_data)
            test_ids_list.append(rid)

    # 8. Handle validation_mode (Subset to 200 rows)
    if validation_mode:
        print("Validation mode: Subsetting to 200 samples.")
        X_train_list = X_train_list[:200]
        y_train_list = y_train_list[:200]
        X_test_list = X_test_list[:min(200, len(X_test_list))]
        test_ids_list = test_ids_list[:min(200, len(test_ids_list))]

    # Final Conversions
    X_train = X_train_list
    y_train = np.array(y_train_list, dtype=np.float32)
    X_test = X_test_list
    test_ids = np.array(test_ids_list, dtype=np.int32)

    print(f"Loading complete. Training: {len(X_train)} samples, Test: {len(X_test)} samples.")
    return X_train, y_train, X_test, test_ids
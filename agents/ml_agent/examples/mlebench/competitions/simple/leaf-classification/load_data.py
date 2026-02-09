import os
import cv2
import numpy as np
import pandas as pd
import cudf
import cupy as cp
import mahotas
from cuml.preprocessing import LabelEncoder
from typing import Tuple, Any
from joblib import Parallel, delayed

# Task-adaptive type definitions using RAPIDS for GPU acceleration
X = cudf.DataFrame      # Feature matrix type: RAPIDS DataFrame
y = cudf.Series         # Target vector type: RAPIDS Series (Label Encoded)
Ids = cudf.Series       # Identifier type: RAPIDS Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/leaf-classification/prepared/public"
OUTPUT_DATA_PATH = "output/5e63fe40-52af-4d8b-ac71-4d3a91b9999f/54/executor/output"

def _extract_augmented_features(image_id: int) -> dict:
    """
    Extracts high-fidelity features: 14 morphological, 25 Zernike, and 20 Fourier descriptors.
    """
    img_path = os.path.join(BASE_DATA_PATH, "images", f"{image_id}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Define keys for all augmented features
    morph_keys = ["m_area", "m_perimeter", "m_solidity", "m_eccentricity", "m_aspect_ratio", "m_extent", "m_circularity"]
    hu_keys = [f"m_hu_{i}" for i in range(7)]
    zernike_keys = [f"z_{i}" for i in range(25)]
    fourier_keys = [f"f_{i}" for i in range(20)]
    all_keys = morph_keys + hu_keys + zernike_keys + fourier_keys
    
    features = {k: 0.0 for k in all_keys}
    features['id'] = image_id
    
    if img is None:
        return features

    # Binary thresholding: Invert so leaf is foreground (255)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return features
    
    # Selection of the largest contour
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return features

    # --- 14 Morphological Features ---
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            _, (MA, ma), _ = cv2.fitEllipse(cnt)
            major_axis, minor_axis = max(MA, ma), min(MA, ma)
            eccentricity = np.sqrt(max(0, 1 - (minor_axis**2 / major_axis**2))) if major_axis > 0 else 0.0
        except: pass
        
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    extent = float(area) / (w * h) if (w * h) > 0 else 0.0
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    
    # 7 Log-Transformed Hu Moments
    hu = cv2.HuMoments(M).flatten()
    hu_log = np.log1p(np.abs(hu))
    
    features.update({
        "m_area": area, "m_perimeter": perimeter, "m_solidity": solidity,
        "m_eccentricity": eccentricity, "m_aspect_ratio": aspect_ratio, 
        "m_extent": extent, "m_circularity": circularity
    })
    for i in range(7):
        features[f"m_hu_{i}"] = hu_log[i]

    # --- 25 Zernike Moments ---
    img_128 = cv2.resize(binary, (128, 128))
    z_moments = mahotas.features.zernike_moments(img_128, radius=64, degree=8)
    for i in range(min(25, len(z_moments))):
        features[f"z_{i}"] = z_moments[i]

    # --- 20 Fourier Descriptors ---
    # Resample contour to 128 points
    cnt_pts = cnt.reshape(-1, 2)
    dists = np.sqrt(np.sum(np.diff(cnt_pts, axis=0, append=cnt_pts[:1])**2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dists[-1]
    
    if total_dist > 0:
        new_dist = np.linspace(0, total_dist, 128, endpoint=False)
        res_x = np.interp(new_dist, cum_dists, np.concatenate((cnt_pts[:, 0], [cnt_pts[0, 0]])))
        res_y = np.interp(new_dist, cum_dists, np.concatenate((cnt_pts[:, 1], [cnt_pts[0, 1]])))
        z_pts = res_x + 1j * res_y
        
        # FFT and extract AC components
        f_coeffs = np.fft.fft(z_pts)
        f_mags = np.abs(f_coeffs[1:21]) # First 20 AC components (index 1 to 20)
        
        # Normalize by fundamental frequency (index 1, which is f_mags[0])
        fundamental = f_mags[0]
        if fundamental > 1e-8:
            f_mags = f_mags / fundamental
        
        # Apply log1p as per specification
        f_mags_log = np.log1p(f_mags)
        for i in range(len(f_mags_log)):
            features[f"f_{i}"] = f_mags_log[i]

    return features

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets with multi-modal feature augmentation.
    """
    print(f"Initializing data loading from {BASE_DATA_PATH}...")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Missing core CSV files in the source directory.")

    train_pd = pd.read_csv(train_path)
    test_pd = pd.read_csv(test_path)
    
    # Feature Augmentation and Caching
    cache_dir = os.path.join(OUTPUT_DATA_PATH, "prepared_data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "augmented_features_v2.csv")
    
    if os.path.exists(cache_path):
        print(f"Loading cached augmented features from {cache_path}")
        aug_features_df = pd.read_csv(cache_path)
    else:
        all_ids = pd.concat([train_pd['id'], test_pd['id']]).unique()
        print(f"Extracting features for {len(all_ids)} images using 36 cores...")
        results = Parallel(n_jobs=36)(delayed(_extract_augmented_features)(idx) for idx in all_ids)
        aug_features_df = pd.DataFrame(results)
        aug_features_df.to_csv(cache_path, index=False)

    # Merge augmented features
    train_full = train_pd.merge(aug_features_df, on='id', how='left')
    test_full = test_pd.merge(aug_features_df, on='id', how='left')

    # Taxonomic Feature: Extract Genus
    train_full['genus'] = train_full['species'].astype(str).apply(lambda x: x.split('_')[0])
    
    # Convert to GPU
    train_gdf = cudf.DataFrame.from_pandas(train_full)
    test_gdf = cudf.DataFrame.from_pandas(test_full)

    # Label Encode Genus
    le_genus = LabelEncoder()
    train_gdf['genus_feat'] = le_genus.fit_transform(train_gdf['genus'])
    
    # For test data, we don't have species, so genus is unknown.
    # We assign a constant value or try to map if possible, here using -1 as a placeholder.
    test_gdf['genus_feat'] = -1

    # Label Encode Target Species
    le_species = LabelEncoder()
    y_train_all = le_species.fit_transform(train_gdf['species'])
    
    # Drop non-feature columns
    # Features = 192 (original) + 14 (morph/hu) + 25 (zernike) + 20 (fourier) + 1 (genus_feat)
    drop_cols_train = ['id', 'species', 'genus']
    drop_cols_test = ['id']
    
    X_train_all = train_gdf.drop(columns=drop_cols_train)
    X_test_all = test_gdf.drop(columns=drop_cols_test)
    test_ids_all = test_gdf['id']

    if validation_mode:
        print("Validation mode enabled: subsetting to 200 samples.")
        X_train = X_train_all.head(200)
        y_train = y_train_all.head(200)
        X_test = X_test_all.head(200)
        test_ids = test_ids_all.head(200)
    else:
        X_train = X_train_all
        y_train = y_train_all
        X_test = X_test_all
        test_ids = test_ids_all

    print(f"Data loading complete. Training features: {X_train.shape[1]}, Training samples: {X_train.shape[0]}")
    return X_train, y_train, X_test, test_ids
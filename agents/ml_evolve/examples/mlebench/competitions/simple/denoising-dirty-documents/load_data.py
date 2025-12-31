# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import glob
import os
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/denoising-dirty-documents/prepared/public"
OUTPUT_DATA_PATH = "output/f88466a1-e032-494a-acbe-a8ee4e4d23cf/9/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets for the Denoising Dirty Documents task.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset for actual training/inference.
            - True: Load a small subset of data (≤50 rows) for quick code validation.

    Returns:
        Tuple[DT, DT, DT, DT]: A tuple containing four elements:
        - X (DT): Training data features (patches of size 112x112x1).
        - y (DT): Training data labels (patches of size 112x112x1).
        - X_test (DT): Test data features (full images with shape HxWx1).
        - test_ids (DT): Identifiers for the test data.
    """
    # -------------------------------------------------------------------------
    # 1. Configuration & Paths
    # -------------------------------------------------------------------------
    train_dir = os.path.join(BASE_DATA_PATH, "train")
    train_cleaned_dir = os.path.join(BASE_DATA_PATH, "train_cleaned")
    test_dir = os.path.join(BASE_DATA_PATH, "test")

    # Hyperparameters for Iteration 6 strategy (High-density patches)
    PATCH_SIZE = 112
    STRIDE = 40  # High overlap to maximize supervision signal

    # Validation constraints
    MAX_ROWS_TRAIN = 50 if validation_mode else float('inf')
    MAX_ROWS_TEST = 50 if validation_mode else float('inf')

    # -------------------------------------------------------------------------
    # 2. Helper Functions
    # -------------------------------------------------------------------------
    def load_image_normalized(path: str) -> np.ndarray:
        """
        Loads an image in grayscale, normalizes to [0, 1], and ensures (H, W, 1) shape.
        """
        if not os.path.exists(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        # Normalize to float [0, 1]
        img = img.astype(np.float32) / 255.0
        # Expand dims to ensure (H, W, 1)
        return img[..., np.newaxis]

    def get_patches(img: np.ndarray, patch_size: int, stride: int) -> list[np.ndarray]:
        """
        Extracts patches from an image (H, W, 1) with REFLECT padding.
        Ensures patches cover the entire image and returns shape (N, P, P, 1).
        """
        h, w = img.shape[:2]

        pad_bottom = 0
        pad_right = 0

        # 1. Ensure dimensions are at least patch_size
        if h < patch_size:
            pad_bottom = patch_size - h
        if w < patch_size:
            pad_right = patch_size - w

        vh = h + pad_bottom
        vw = w + pad_right

        # 2. Ensure dimensions allow for exact striding
        if (vh - patch_size) % stride != 0:
            pad_bottom += stride - ((vh - patch_size) % stride)
        if (vw - patch_size) % stride != 0:
            pad_right += stride - ((vw - patch_size) % stride)

        # Apply padding if needed
        if pad_bottom > 0 or pad_right > 0:
            # copyMakeBorder works on (H, W, C) properly for recent OpenCV versions
            img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
            # Ensure shape is maintained (OpenCV sometimes drops singleton dim on border ops)
            if len(img.shape) == 2:
                img = img[..., np.newaxis]

        new_h, new_w = img.shape[:2]
        patches = []

        # Extract patches
        for y in range(0, new_h - patch_size + 1, stride):
            for x in range(0, new_w - patch_size + 1, stride):
                # Slice maintains channel dimension (112, 112, 1)
                patch = img[y: y + patch_size, x: x + patch_size, :]
                patches.append(patch)

        return patches

    # -------------------------------------------------------------------------
    # 3. Load Training Data (X, y)
    # -------------------------------------------------------------------------
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.png")))
    train_cleaned_files = sorted(glob.glob(os.path.join(train_cleaned_dir, "*.png")))

    # Optimization: Restrict input files in validation mode to avoid excess I/O.
    # Stride 40 produces many patches per image, so 1-2 images are sufficient.
    if validation_mode:
        train_files = train_files[:2]
        train_cleaned_files = train_cleaned_files[:2]

    X_rows = []
    y_rows = []

    for f_dirty, f_clean in zip(train_files, train_cleaned_files):
        if len(X_rows) >= MAX_ROWS_TRAIN:
            break

        img_id = os.path.basename(f_dirty).split('.')[0]

        img_dirty = load_image_normalized(f_dirty)
        img_clean = load_image_normalized(f_clean)

        if img_dirty is None or img_clean is None:
            continue

        # Robustness: Handle size mismatch if any (resize clean to match dirty)
        if img_dirty.shape != img_clean.shape:
            h, w = img_dirty.shape[:2]
            # cv2.resize expects (W, H)
            img_clean_resized = cv2.resize(img_clean, (w, h))
            img_clean = img_clean_resized[..., np.newaxis]

        # Create patches (112, 112, 1)
        patches_dirty = get_patches(img_dirty, PATCH_SIZE, STRIDE)
        patches_clean = get_patches(img_clean, PATCH_SIZE, STRIDE)

        for pd_img, pc_img in zip(patches_dirty, patches_clean):
            X_rows.append({'image_id': img_id, 'data': pd_img})
            y_rows.append({'label': pc_img})

            if len(X_rows) >= MAX_ROWS_TRAIN:
                break

    X = pd.DataFrame(X_rows)
    y = pd.DataFrame(y_rows)

    # -------------------------------------------------------------------------
    # 4. Load Test Data (X_test, test_ids)
    # -------------------------------------------------------------------------
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.png")))

    X_test_rows = []
    test_ids_rows = []

    for f_test in test_files:
        if len(X_test_rows) >= MAX_ROWS_TEST:
            break

        img_id = os.path.basename(f_test).split('.')[0]
        img_test = load_image_normalized(f_test)

        if img_test is None:
            continue

        # X_test stores full images with (H, W, 1) shape for full context prediction
        X_test_rows.append({'image_id': img_id, 'data': img_test})
        test_ids_rows.append(img_id)

    X_test = pd.DataFrame(X_test_rows)
    test_ids = pd.DataFrame(test_ids_rows, columns=['image_id'])

    # -------------------------------------------------------------------------
    # 5. Final Structure Enforcement
    # -------------------------------------------------------------------------
    # Ensure columns exist even if empty
    if 'image_id' not in X.columns: X['image_id'] = pd.Series(dtype='object')
    if 'data' not in X.columns: X['data'] = pd.Series(dtype='object')

    if 'label' not in y.columns: y['label'] = pd.Series(dtype='object')

    if 'image_id' not in X_test.columns: X_test['image_id'] = pd.Series(dtype='object')
    if 'data' not in X_test.columns: X_test['data'] = pd.Series(dtype='object')

    # Enforce strict column order
    X = X[['image_id', 'data']]
    y = y[['label']]
    X_test = X_test[['image_id', 'data']]

    return X, y, X_test, test_ids

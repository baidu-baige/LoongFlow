# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import io
import os
import zipfile
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/aerial-cactus-identification/prepared/public"
OUTPUT_DATA_PATH = "output/153e4624-940b-4d19-a37d-90435531bfd1/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data() -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets.

    This function takes no arguments as it should derive file paths from the task description
    or predefined global variables.

    Returns:
        Tuple[DT, DT, DT, DT]: A tuple containing four elements:
        - X (DT): Training data features.
        - y (DT): Training data labels.
        - X_test (DT): Test data features.
        - test_ids (DT): Identifiers for the test data.
    """
    # Define file paths
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    train_zip_path = os.path.join(BASE_DATA_PATH, "train.zip")
    test_zip_path = os.path.join(BASE_DATA_PATH, "test.zip")

    # Load training labels
    train_df = pd.read_csv(train_csv_path)

    # Create a dictionary mapping image id to label
    label_dict = dict(zip(train_df['id'], train_df['has_cactus']))

    # Load training images from zip
    train_images = []
    train_labels = []
    train_ids = []

    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        # Get list of image files in the zip
        image_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')]

        for img_file in image_files:
            # Extract the filename (without directory path if any)
            img_name = os.path.basename(img_file)

            if img_name in label_dict:
                # Read image from zip
                with zip_ref.open(img_file) as f:
                    img_data = f.read()
                    img = Image.open(io.BytesIO(img_data))
                    img_array = np.array(img)

                    # Normalize to [0, 1] range
                    img_array = img_array.astype(np.float32) / 255.0

                    # Flatten the image for DataFrame storage
                    # Original shape: (32, 32, 3) -> flattened: (3072,)
                    img_flat = img_array.flatten()

                    train_images.append(img_flat)
                    train_labels.append(label_dict[img_name])
                    train_ids.append(img_name)

    # Load test images from zip
    test_images = []
    test_ids_list = []

    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        # Get list of image files in the zip
        image_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')]

        for img_file in image_files:
            # Extract the filename (without directory path if any)
            img_name = os.path.basename(img_file)

            if img_name:  # Skip empty names (directories)
                # Read image from zip
                with zip_ref.open(img_file) as f:
                    img_data = f.read()
                    img = Image.open(io.BytesIO(img_data))
                    img_array = np.array(img)

                    # Normalize to [0, 1] range
                    img_array = img_array.astype(np.float32) / 255.0

                    # Flatten the image for DataFrame storage
                    img_flat = img_array.flatten()

                    test_images.append(img_flat)
                    test_ids_list.append(img_name)

    # Create column names for the flattened image data
    # 32x32x3 = 3072 features
    num_features = 32 * 32 * 3
    column_names = [f'pixel_{i}' for i in range(num_features)]

    # Create DataFrames
    X = pd.DataFrame(train_images, columns=column_names)
    y = pd.Series(train_labels, name='has_cactus')
    X_test = pd.DataFrame(test_images, columns=column_names)
    test_ids = pd.Series(test_ids_list, name='id')

    return X, y, X_test, test_ids

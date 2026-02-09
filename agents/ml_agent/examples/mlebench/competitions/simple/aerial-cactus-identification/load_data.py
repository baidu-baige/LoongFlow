import pandas as pd
import os
import zipfile
from typing import Tuple

# Task-adaptive type definitions
X = pd.Series  # Feature matrix: Series of absolute image paths
y = pd.Series  # Target vector: Series of binary labels (int)
Ids = pd.Series  # Identifier type: Series of image filenames (strings)

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/aerial-cactus-identification/prepared/public"
OUTPUT_DATA_PATH = "output/eac64466-fa75-4fb8-ade1-75e2091ddf4a/1/executor/output"


def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the Aerial Cactus Identification dataset.

    Args:
        validation_mode: If True, returns at most 200 representative samples.

    Returns:
        X_train: pd.Series of absolute file paths for training images.
        y_train: pd.Series of labels (0 or 1).
        X_test: pd.Series of absolute file paths for test images.
        test_ids: pd.Series of IDs for submission file creation.
    """
    # Step 0: Ensure data readiness (Extraction)
    # We use a named subdirectory to store the extracted images.
    # Note: Preparation is always full, independent of validation_mode.
    extract_base = os.path.join(BASE_DATA_PATH, "extracted_images")

    for zip_name in ["train.zip", "test.zip"]:
        zip_path = os.path.join(BASE_DATA_PATH, zip_name)
        folder_name = zip_name.replace(".zip", "")
        target_dir = os.path.join(extract_base, folder_name)

        # Verify if extraction is needed (check if folder exists and is non-empty)
        if not os.path.exists(target_dir) or not os.listdir(target_dir):
            print(f"Preparing data: Extracting {zip_path} to {extract_base}...")
            os.makedirs(extract_base, exist_ok=True)

            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Missing required archive: {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_base)

            # Post-extraction verification
            print(f"Finished extracting {zip_name}.")

    # Step 1: Load metadata
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError("Required CSV metadata files are missing.")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Step 2: Apply validation_mode subsetting
    if validation_mode:
        print("Validation mode: Sampling 200 representative rows from train and test.")
        # Sample with fixed random state for reproducibility
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=42)

    # Step 3: Resolve absolute paths and structure returns
    def resolve_absolute_path(img_id, subset):
        """
        Resolves the absolute path of an image, handling different possible zip structures.
        """
        # Candidate 1: Standard extraction (extract_base/subset/img_id)
        path_std = os.path.join(extract_base, subset, img_id)
        if os.path.exists(path_std):
            return path_std

        # Candidate 2: Flat extraction (extract_base/img_id)
        path_flat = os.path.join(extract_base, img_id)
        if os.path.exists(path_flat):
            return path_flat

        # Candidate 3: Doubly nested (extract_base/subset/subset/img_id)
        path_nested = os.path.join(extract_base, subset, subset, img_id)
        if os.path.exists(path_nested):
            return path_nested

        raise FileNotFoundError(f"Image ID '{img_id}' not found in prepared directory '{extract_base}'.")

    print("Mapping image paths and finalising datasets...")
    X_train_paths = train_df['id'].apply(lambda x: resolve_absolute_path(x, "train")).reset_index(drop=True)
    y_train_labels = train_df['has_cactus'].reset_index(drop=True)

    X_test_paths = test_df['id'].apply(lambda x: resolve_absolute_path(x, "test")).reset_index(drop=True)
    test_ids_out = test_df['id'].reset_index(drop=True)

    print(f"Data loading complete. Train size: {len(X_train_paths)}, Test size: {len(X_test_paths)}")

    return X_train_paths, y_train_labels, X_test_paths, test_ids_out
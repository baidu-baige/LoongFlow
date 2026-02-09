from typing import Tuple
import pandas as pd
import os
import subprocess

# Paths defined in the environment
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/iwildcam-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/939424a9-5a07-4c99-9f56-709187b5b05c/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series
Ids = pd.Series

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the iWildCam 2019 datasets, extracting images and metadata.

    Args:
        validation_mode: If True, returns at most 200 rows for training and testing.

    Returns:
        Tuple[X_train, y_train, X_test, test_ids]
    """
    print("Initializing data loading process...")

    # Define source and destination paths
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")
    train_zip = os.path.join(BASE_DATA_PATH, "train_images.zip")
    test_zip = os.path.join(BASE_DATA_PATH, "test_images.zip")
    
    # Preparation: Extract images to specific subdirectories
    train_img_dir = os.path.join(BASE_DATA_PATH, "train_images_extracted")
    test_img_dir = os.path.join(BASE_DATA_PATH, "test_images_extracted")

    def ensure_data_prepared(zip_path: str, target_dir: str):
        """Extracts zip files if the target directory does not exist or is empty."""
        if not os.path.exists(target_dir) or not os.listdir(target_dir):
            print(f"Extracting {zip_path} to {target_dir}...")
            os.makedirs(target_dir, exist_ok=True)
            try:
                # Use subprocess with -q (quiet) for efficient extraction
                subprocess.run(["unzip", "-q", zip_path, "-d", target_dir], check=True)
                print(f"Successfully extracted {os.path.basename(zip_path)}.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract {zip_path}: {e}")
                raise e
        else:
            print(f"Using existing data in {target_dir}.")

    # Data preparation must always be full, independent of validation_mode
    ensure_data_prepared(train_zip, train_img_dir)
    ensure_data_prepared(test_zip, test_img_dir)

    # Step 1: Load raw CSV data
    print("Loading metadata from CSV files...")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Step 2: Determine image path mapping structure
    # Some zips contain a nested folder with the same name as the zip
    def get_path_prefix(img_dir: str, df: pd.DataFrame) -> str:
        sample_file = df['file_name'].iloc[0]
        if os.path.exists(os.path.join(img_dir, sample_file)):
            return ""
        # Check common subfolder pattern (e.g., train_images/xxx.jpg)
        subfolder = os.path.basename(img_dir).replace('_extracted', '')
        if os.path.exists(os.path.join(img_dir, subfolder, sample_file)):
            return subfolder
        return ""

    train_prefix = get_path_prefix(train_img_dir, train_df)
    test_prefix = get_path_prefix(test_img_dir, test_df)

    train_df['image_path'] = train_df['file_name'].apply(lambda x: os.path.join(train_img_dir, train_prefix, x))
    test_df['image_path'] = test_df['file_name'].apply(lambda x: os.path.join(test_img_dir, test_prefix, x))

    # Step 3: Normalize metadata features (location, seq_num_frames, frame_num)
    # These features have high correlation with the target or provide crucial context
    print("Normalizing predictive metadata features...")
    numeric_meta_cols = ['location', 'seq_num_frames', 'frame_num']
    for col in numeric_meta_cols:
        train_df[col] = train_df[col].astype('float32')
        test_df[col] = test_df[col].astype('float32')
        
        # Calculate min-max based on combined distribution for consistency
        global_min = min(train_df[col].min(), test_df[col].min())
        global_max = max(train_df[col].max(), test_df[col].max())
        
        span = global_max - global_min
        if span > 0:
            train_df[col] = (train_df[col] - global_min) / span
            test_df[col] = (test_df[col] - global_min) / span
        else:
            train_df[col] = 0.0
            test_df[col] = 0.0

    # Step 4: Apply validation_mode subsetting
    if validation_mode:
        print("Validation mode enabled: subsetting to 200 samples.")
        train_df = train_df.head(200).reset_index(drop=True)
        test_df = test_df.head(200).reset_index(drop=True)

    # Step 5: Structure and return data
    # X includes image paths and normalized metadata; y is the target; Ids for submission mapping
    feature_cols = ['image_path', 'location', 'seq_id', 'seq_num_frames', 'frame_num']
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['category_id'].copy()
    X_test = test_df[feature_cols].copy()
    test_ids = test_df['id'].copy()

    print(f"Data loading complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, test_ids
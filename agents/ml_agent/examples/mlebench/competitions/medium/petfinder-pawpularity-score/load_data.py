import pandas as pd
import os
from typing import Tuple, List

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/petfinder-pawpularity-score/prepared/public"
OUTPUT_DATA_PATH = "output/10daf1c7-eb8a-49b6-bcf3-ba43767dcbe6/2/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing image paths and metadata
y = pd.DataFrame      # Target vector containing scaled Pawpularity and metadata labels
Ids = pd.Series       # Identifier series for output alignment

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the PetFinder Pawpularity dataset.
    
    The function loads tabular metadata and constructs image paths. 
    Targets are scaled (Pawpularity/100) and include metadata for multi-task learning.
    """
    print(f"Loading data from {BASE_DATA_PATH}...")
    
    # Define file paths
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")
    train_img_dir = os.path.join(BASE_DATA_PATH, "train")
    test_img_dir = os.path.join(BASE_DATA_PATH, "test")

    # Load CSV files
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # The EDA report uses 'Subject Focus' while the description uses 'Focus'.
    # We will detect the actual column names from the CSV to ensure robustness.
    metadata_candidates = [
        'Subject Focus', 'Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ]
    metadata_cols = [col for col in metadata_candidates if col in train_df.columns]
    
    print(f"Detected metadata columns: {metadata_cols}")
    if len(metadata_cols) != 12:
        print(f"Warning: Expected 12 metadata columns, but found {len(metadata_cols)}: {metadata_cols}")

    # Construct image paths
    train_df['image_path'] = train_df['Id'].apply(lambda x: os.path.join(train_img_dir, f"{x}.jpg"))
    test_df['image_path'] = test_df['Id'].apply(lambda x: os.path.join(test_img_dir, f"{x}.jpg"))

    # Verify a sample of image paths exist to ensure directory structure alignment
    sample_path = train_df['image_path'].iloc[0]
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Image path verification failed. Sample path does not exist: {sample_path}")

    # Scale Pawpularity target to [0, 1]
    train_df['Pawpularity_scaled'] = train_df['Pawpularity'].astype('float32') / 100.0

    # Apply validation_mode subsetting (at most 200 rows)
    if validation_mode:
        print("Validation mode enabled: subsetting data to 200 rows.")
        train_df = train_df.head(200).copy()
        test_df = test_df.head(200).copy()

    # Prepare return components
    # X_train: Image paths and metadata features
    X_train = train_df[['image_path'] + metadata_cols].copy()
    for col in metadata_cols:
        X_train[col] = X_train[col].astype('float32')

    # y_train: Multi-task targets (Scaled Pawpularity + Metadata labels)
    y_train = train_df[['Pawpularity_scaled'] + metadata_cols].copy()
    for col in metadata_cols:
        y_train[col] = y_train[col].astype('float32')
        
    # X_test: Image paths and metadata features for inference
    X_test = test_df[['image_path'] + metadata_cols].copy()
    for col in metadata_cols:
        X_test[col] = X_test[col].astype('float32')

    # test_ids: Original Ids for submission alignment
    test_ids = test_df['Id'].copy()

    print(f"Successfully loaded {len(X_train)} training samples and {len(X_test)} test samples.")
    
    return X_train, y_train, X_test, test_ids
import os
import pandas as pd
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/502e395f-5bca-4b30-9a81-fc7b49a1e544/3/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix: Contains 'image_path'
y = pd.Series     # Target vector: Contains 'label'
Ids = pd.Series   # Identifier type: Contains 'image_id'

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the Cassava Leaf Disease Classification task.

    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset.
            - True: Return a small subset of data (200 rows) for quick code validation.

    Returns:
        Tuple[X, y, X, Ids]: (X_train, y_train, X_test, test_ids)
    """
    print(f"Loading data from: {BASE_DATA_PATH}")

    # Step 1: Load metadata from CSV sources
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training metadata not found at {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata (sample submission) not found at {test_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Step 2: Construct absolute file paths for images
    # We map the base directory with image_id to ensure downstream stages can access files directly.
    train_df['image_path'] = train_df['image_id'].apply(
        lambda x: os.path.join(BASE_DATA_PATH, "train_images", x)
    )
    test_df['image_path'] = test_df['image_id'].apply(
        lambda x: os.path.join(BASE_DATA_PATH, "test_images", x)
    )

    # Step 3: Validate physical existence of image files (sampling-based check for efficiency)
    print("Validating image paths on filesystem...")
    
    # Check a sample of 10 images to ensure the structure is correct
    sample_train_paths = train_df['image_path'].sample(min(10, len(train_df)), random_state=42)
    for path in sample_train_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image validation failed. Path does not exist: {path}")
            
    sample_test_paths = test_df['image_path'].sample(min(10, len(test_df)), random_state=42)
    for path in sample_test_paths:
        if not os.path.exists(path):
            # Note: In some hidden-test scenarios, images might only exist in the scoring environment.
            # However, based on the task description and EDA, test images are present in the provided path.
            raise FileNotFoundError(f"Test image validation failed. Path does not exist: {path}")

    # Step 4: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Validation mode enabled: Subsetting data to 200 samples.")
        # Taking a representative subset by using a fixed seed
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=42).reset_index(drop=True)

    # Step 5: Structure into required return format
    X_train = train_df[['image_path']].copy()
    y_train = train_df['label'].copy()
    
    X_test = test_df[['image_path']].copy()
    test_ids = test_df['image_id'].copy()

    # Final validation of row alignment and non-emptiness
    assert not X_train.empty, "X_train must not be empty"
    assert not y_train.empty, "y_train must not be empty"
    assert not X_test.empty, "X_test must not be empty"
    assert not test_ids.empty, "test_ids must not be empty"
    
    assert len(X_train) == len(y_train), "Mismatch between training features and targets"
    assert len(X_test) == len(test_ids), "Mismatch between test features and identifiers"

    print(f"Successfully loaded {len(X_train)} training samples and {len(X_test)} test samples.")

    return X_train, y_train, X_test, test_ids
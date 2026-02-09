from typing import Tuple
import os
import re
import pandas as pd

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/25e7371d-bfe6-47c9-b200-bdf664ef9932/2/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame   # Feature matrix type (contains paths and metadata)
y = pd.Series      # Target vector type (binary labels)
Ids = pd.Series    # Identifier type for output alignment (numeric IDs)

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the Dogs vs. Cats classification task.

    Args:
        validation_mode: Controls the data loading behavior.
            - False: Load the complete dataset.
            - True: Return a small subset (200 rows) for quick code validation.
    Returns:
        Tuple[X, y, X, Ids]: 
        - X_train: Training metadata (path, id)
        - y_train: Training labels (0 for cat, 1 for dog)
        - X_test: Test metadata (path, id)
        - test_ids: Test identifiers for submission
    """
    train_dir = os.path.join(BASE_DATA_PATH, "train")
    test_dir = os.path.join(BASE_DATA_PATH, "test")

    # Step 0: Ensure data readiness
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Required data directories 'train' or 'test' not found in {BASE_DATA_PATH}")

    label_mapping = {'cat': 0, 'dog': 1}
    
    # Step 1: Load training data using regex-based filename parsing
    print(f"Loading training data from: {train_dir}")
    train_records = []
    for filename in os.listdir(train_dir):
        if filename.lower().endswith('.jpg'):
            # Expected format: label.id.jpg (e.g., cat.1.jpg)
            match = re.match(r'^(cat|dog)\.(\d+)\.jpg$', filename, re.IGNORECASE)
            if match:
                label_str, img_id = match.groups()
                train_records.append({
                    'path': os.path.abspath(os.path.join(train_dir, filename)),
                    'label': label_mapping[label_str.lower()],
                    'id': int(img_id)
                })
    
    df_train = pd.DataFrame(train_records)
    if df_train.empty:
        raise ValueError(f"No valid training images found in {train_dir} matching 'label.id.jpg' pattern.")
    
    # Ensure deterministic order
    df_train = df_train.sort_values('id').reset_index(drop=True)

    # Step 2: Load test data using regex-based filename parsing
    print(f"Loading test data from: {test_dir}")
    test_records = []
    for filename in os.listdir(test_dir):
        if filename.lower().endswith('.jpg'):
            # Expected format: id.jpg (e.g., 1.jpg)
            match = re.match(r'^(\d+)\.jpg$', filename)
            if match:
                img_id = match.groups()[0]
                test_records.append({
                    'path': os.path.abspath(os.path.join(test_dir, filename)),
                    'id': int(img_id)
                })
    
    df_test = pd.DataFrame(test_records)
    if df_test.empty:
        raise ValueError(f"No valid test images found in {test_dir} matching 'id.jpg' pattern.")
    
    # Ensure deterministic order
    df_test = df_test.sort_values('id').reset_index(drop=True)

    # Step 3: Apply validation_mode subsetting if enabled
    if validation_mode:
        print("Validation mode enabled: Sampling 200 rows for train and test.")
        df_train = df_train.sample(n=min(len(df_train), 200), random_state=42).reset_index(drop=True)
        df_test = df_test.sample(n=min(len(df_test), 200), random_state=42).reset_index(drop=True)

    # Step 4: Verify path existence for a sample to ensure alignment
    for df_sample, name in [(df_train, "Train"), (df_test, "Test")]:
        if not df_sample.empty:
            check_path = df_sample.iloc[0]['path']
            if not os.path.exists(check_path):
                raise FileNotFoundError(f"Critical Error: {name} file path {check_path} does not exist.")

    # Structure return values
    X_train = df_train[['path', 'id']]
    y_train = df_train['label']
    X_test = df_test[['path', 'id']]
    test_ids = df_test['id']

    print(f"Successfully loaded {len(X_train)} training samples and {len(X_test)} test samples.")
    
    return X_train, y_train, X_test, test_ids
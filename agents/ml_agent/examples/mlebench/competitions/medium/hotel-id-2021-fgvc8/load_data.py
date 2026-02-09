import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/hotel-id-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/6c275358-248b-46e3-a3f8-feb17fef7b7f/3/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Features: contains image_path, hotel_id, and group_id
y = pd.Series     # Target: encoded hotel_id
Ids = pd.Series   # Identifier: image filenames for submission

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the Hotel Recognition task.
    Constructs image paths, encodes targets, and generates temporal group IDs for CV.
    """
    print("Stage 1: Loading metadata and preparing datasets...")
    
    # 1. Load raw metadata
    train_csv_path = os.path.join(BASE_DATA_PATH, 'train.csv')
    sample_sub_path = os.path.join(BASE_DATA_PATH, 'sample_submission.csv')
    
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Missing training metadata: {train_csv_path}")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"Missing sample submission: {sample_sub_path}")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(sample_sub_path)
    
    # 2. Construct absolute paths
    # Training images are nested: train_images/<chain>/<image>
    train_df['image_path'] = train_df.apply(
        lambda row: os.path.join(BASE_DATA_PATH, 'train_images', str(row['chain']), row['image']), 
        axis=1
    )
    
    # Test images are flat: test_images/<image>
    test_df['image_path'] = test_df['image'].apply(
        lambda img: os.path.join(BASE_DATA_PATH, 'test_images', img)
    )
    
    # 3. Handle 87798 vs 87797 discrepancy (Missing file alignment)
    print("Verifying image existence and aligning with filesystem...")
    # Sample verification for fast failure
    sample_paths = train_df['image_path'].head(10).tolist()
    for p in sample_paths:
        if not os.path.exists(p):
            # If the first few fail, the path logic itself is likely wrong
            raise FileNotFoundError(f"Critical path logic error. Sample not found: {p}")

    # Full verification to catch the single missing image identified in EDA
    exists_mask = [os.path.exists(p) for p in train_df['image_path']]
    num_missing = len(train_df) - sum(exists_mask)
    if num_missing > 0:
        print(f"Filtering {num_missing} missing image(s) from metadata.")
        train_df = train_df[exists_mask].reset_index(drop=True)

    # 4. Target Label Encoding
    print("Encoding targets...")
    le_hotel = LabelEncoder()
    # We store the raw hotel_id in the dataframe but use the encoded version for y
    train_df['hotel_id_encoded'] = le_hotel.fit_transform(train_df['hotel_id'])
    
    # Save mapping for inference/ensemble stages
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    mapping_df = pd.DataFrame({
        'hotel_id': le_hotel.classes_,
        'encoded_label': range(len(le_hotel.classes_))
    })
    mapping_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'hotel_id_mapping.csv'), index=False)

    # 5. Create temporal Group IDs
    # Combining hotel_id and timestamp to ensure images from the same "incident" stay together.
    # We use ngroup() to create a unique integer identifier for each (hotel, timestamp) pair.
    print("Creating temporal group IDs...")
    train_df['group_id'] = train_df.groupby(['hotel_id', 'timestamp']).ngroup()

    # 6. Apply Validation Subsetting
    if validation_mode:
        print("Validation mode: Subsetting to 200 samples.")
        # Use random_state for reproducibility
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=42).reset_index(drop=True)

    # 7. Structure Returns
    # X_train requires: image_path, hotel_id, group_id
    X_train = train_df[['image_path', 'hotel_id', 'group_id']]
    # y_train is the encoded target
    y_train = train_df['hotel_id_encoded']
    
    # X_test and Ids
    X_test = test_df[['image_path']]
    test_ids = test_df['image']
    
    print(f"Initial load complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Unique hotels: {len(le_hotel.classes_)}")
    
    return X_train, y_train, X_test, test_ids
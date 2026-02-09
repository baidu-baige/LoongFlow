import os
import json
import pandas as pd
from typing import Tuple, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/iwildcam-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/9508c267-92c9-4fd0-91c8-90efc0fba263/1/executor/output"

# Concrete type definitions for this task
# X: pd.DataFrame with columns ['file_name', 'location', 'seq_id', 'bboxes']
# y: pd.Series containing 'category_id'
# Ids: pd.Series containing image 'id' for alignment
X = pd.DataFrame
y = pd.Series
Ids = pd.Series

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the iWildCam 2020 task.
    Combines image metadata, annotations, and MegaDetector bounding box results.
    """
    print("Starting data loading process...")

    # Define file paths
    train_json_path = os.path.join(BASE_DATA_PATH, "iwildcam2020_train_annotations.json")
    test_json_path = os.path.join(BASE_DATA_PATH, "iwildcam2020_test_information.json")
    md_json_path = os.path.join(BASE_DATA_PATH, "iwildcam2020_megadetector_results.json")

    # Verify input files exist
    for path in [train_json_path, test_json_path, md_json_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")

    # Step 1: Load MegaDetector results
    print("Loading MegaDetector results...")
    with open(md_json_path, 'r') as f:
        md_raw = json.load(f)
    
    # Map image ID to detections (bboxes)
    # MegaDetector results often contain multiple categories (1=animal, 2=person, 3=vehicle)
    # We prioritize animal detections but fallback to others if requested or default to full image
    md_map = {}
    for img_entry in md_raw.get('images', []):
        img_id = img_entry.get('id')
        detections = img_entry.get('detections', [])
        
        # Extract bboxes for all detections (following "top 100 boxes" spec)
        # BBox format in this dataset is typically relative [ymin, xmin, ymax, xmax]
        bboxes = [d['bbox'] for d in detections]
        
        if not bboxes:
            bboxes = [[0.0, 0.0, 1.0, 1.0]] # Default to full image if no detection
            
        md_map[img_id] = bboxes

    # Step 2: Load and structure Train data
    print("Loading training annotations...")
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    train_images_df = pd.DataFrame(train_data['images'])
    train_ann_df = pd.DataFrame(train_data['annotations'])
    
    # Merge annotations (category_id) onto image metadata
    # The 'id' in images matches 'image_id' in annotations
    train_df = pd.merge(
        train_images_df, 
        train_ann_df[['image_id', 'category_id']], 
        left_on='id', 
        right_on='image_id', 
        how='inner'
    )
    
    # Step 3: Load and structure Test data
    print("Loading test information...")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    test_df = pd.DataFrame(test_data['images'])

    # Step 4: Map MegaDetector boxes to both dataframes
    print("Mapping MegaDetector boxes...")
    train_df['bboxes'] = train_df['id'].map(md_map).apply(lambda x: x if isinstance(x, list) else [[0.0, 0.0, 1.0, 1.0]])
    test_df['bboxes'] = test_df['id'].map(md_map).apply(lambda x: x if isinstance(x, list) else [[0.0, 0.0, 1.0, 1.0]])

    # Step 5: Path alignment and verification
    # Prepend directory names to file_name
    train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join("train", x))
    test_df['file_name'] = test_df['file_name'].apply(lambda x: os.path.join("test", x))

    # Verification sampling
    sample_train_path = os.path.join(BASE_DATA_PATH, train_df.iloc[0]['file_name'])
    if not os.path.exists(sample_train_path):
        # If absolute paths are needed or structure is different, this would error out
        # allowing for diagnosis as per requirement 4.
        raise FileNotFoundError(f"Verification failed: Sample image path does not exist: {sample_train_path}")

    # Step 6: Handle validation mode
    if validation_mode:
        print("Validation mode enabled: subsetting data to 200 rows.")
        train_df = train_df.head(200)
        test_df = test_df.head(200)

    # Step 7: Final selection of features and targets
    # Required columns: file_name, location, seq_id, bboxes
    X_train = train_df[['file_name', 'location', 'seq_id', 'bboxes']]
    y_train = train_df['category_id']
    X_test = test_df[['file_name', 'location', 'seq_id', 'bboxes']]
    test_ids = test_df['id']

    print(f"Data loading complete. Train: {len(X_train)} samples, Test: {len(X_test)} samples.")
    
    return X_train, y_train, X_test, test_ids
import pandas as pd
import os
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix containing 'anchor', 'target', 'context', and 'context_desc'
y = pd.Series     # Target vector containing the 'score'
Ids = pd.Series   # Identifier vector containing the 'id' for submission alignment

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/02d42284-9bf3-4f97-ab6c-7ea839095b54/3/executor/output"

# CPC Section mapping based on technical specification to provide semantic signal
CPC_MAPPING = {
    'A': 'Human Necessities',
    'B': 'Performing Operations; Transporting',
    'C': 'Chemistry; Metallurgy',
    'D': 'Textiles; Paper',
    'E': 'Fixed Constructions',
    'F': 'Mechanical Engineering; Lighting; Heating; Weapons; Blasting',
    'G': 'Physics',
    'H': 'Electricity',
    'Y': 'General tagging of new technological developments'
}

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets for the US Patent Phrase Matching task.
    
    This function implements:
    1. Loading of raw CSV files (train and test).
    2. Mapping of CPC context codes to natural language section descriptions.
    3. Persisting prepared data to disk for pipeline consistency.
    4. Representative subsetting for validation mode.
    """
    print(f"Execution: load_data (validation_mode={validation_mode})")
    
    # Path verification
    train_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_path = os.path.join(BASE_DATA_PATH, "test.csv")
    
    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file missing: {path}")

    # Step 1: Load Data
    print(f"Reading CSV files from {BASE_DATA_PATH}...")
    train_df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')
    
    # Step 2: Full Data Preparation (Independent of validation_mode)
    # Augment with CPC context descriptions as per technical specification
    def augment_cpc_features(df: pd.DataFrame) -> pd.DataFrame:
        # Map the first character (Section) of the context code to its description
        # We also ensure the full context code is preserved as 'context'
        df['context_desc'] = df['context'].str[0].map(CPC_MAPPING)
        return df

    print("Augmenting features with CPC context descriptions...")
    train_df = augment_cpc_features(train_df)
    test_df = augment_cpc_features(test_df)

    # Persist prepared data to the specified output directory for downstream stages
    prep_dir = os.path.join(OUTPUT_DATA_PATH, "prepared_data")
    os.makedirs(prep_dir, exist_ok=True)
    
    train_prep_path = os.path.join(prep_dir, "train_prepared.csv")
    test_prep_path = os.path.join(prep_dir, "test_prepared.csv")
    
    # Save prepared data (full version)
    train_df.to_csv(train_prep_path, index=False)
    test_df.to_csv(test_prep_path, index=False)
    print(f"Prepared full datasets saved to: {prep_dir}")

    # Step 3: Validation Mode Subsetting
    # When validation_mode is True, return at most 200 rows
    if validation_mode:
        print("Validation mode enabled: subsetting data to 200 rows.")
        # Taking a representative head; since data order is typically random/shuffled in these CSVs
        train_output_df = train_df.head(200).copy()
        test_output_df = test_df.head(200).copy()
    else:
        train_output_df = train_df
        test_output_df = test_df

    # Step 4: Structure Data into Return Format
    # X: Features (anchor, target, context, context_desc)
    # y: Target (score)
    # Ids: Identifiers (id)
    
    feature_columns = ['anchor', 'target', 'context', 'context_desc']
    
    X_train = train_output_df[feature_columns]
    y_train = train_output_df['score']
    
    X_test = test_output_df[feature_columns]
    test_ids = test_output_df['id']

    # Integrity Checks
    if X_train.empty or y_train.empty or X_test.empty or test_ids.empty:
        raise ValueError("Loading failed: One or more returned structures are empty.")
    
    if len(X_train) != len(y_train):
        raise ValueError(f"Train alignment mismatch: X_train({len(X_train)}) vs y_train({len(y_train)})")
        
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test alignment mismatch: X_test({len(X_test)}) vs test_ids({len(test_ids)})")

    print(f"Successfully loaded data.")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, test_ids
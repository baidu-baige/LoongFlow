import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

# Path constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/3c8d5cca-ccb7-4c25-92f1-7d0f571dedc1/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing tabular and structural data
y = pd.DataFrame      # Target vector for formation energy and bandgap
Ids = pd.Series       # Identifiers for test set alignment

def parse_xyz_file(file_path: str):
    """
    Parses a NOMAD2018 geometry.xyz file.
    Extracts lattice vectors, 3D coordinates, and atomic species.
    """
    lattice = []
    coords = []
    species = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'lattice_vector':
                    lattice.append([float(x) for x in parts[1:]])
                elif parts[0] == 'atom':
                    # Format: atom x y z symbol
                    coords.append([float(x) for x in parts[1:4]])
                    species.append(parts[4])
    except Exception as e:
        # Propagate error as per requirement
        raise RuntimeError(f"Critical error parsing XYZ file at {file_path}: {e}")
        
    return {
        'lattice': np.array(lattice, dtype=np.float32),
        'coords': np.array(coords, dtype=np.float32),
        'species': species
    }

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the NOMAD2018 dataset.
    Combines tabular features from CSV files with atomic structures from XYZ files.
    """
    print(f"Initializing data load from {BASE_DATA_PATH}...")
    
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")
    
    # Verify core files exist
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError("Primary data files (train.csv/test.csv) are missing.")
        
    # Load tabular data
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Target columns as specified in the task
    target_cols = ['formation_energy_ev_natom', 'bandgap_energy_ev']
    
    # Apply validation mode subsetting
    if validation_mode:
        train_df = train_df.head(200).copy()
        test_df = test_df.head(200).copy()
        print(f"Validation mode enabled: Loading subset of 200 rows.")

    def enrich_with_structural_data(df, subset_name):
        """Helper to parse and attach XYZ data to the dataframe in parallel."""
        print(f"Loading structural geometry for {subset_name} subset...")
        
        # Construct paths for each ID
        xyz_paths = [
            os.path.join(BASE_DATA_PATH, subset_name, str(int(row['id'])), "geometry.xyz")
            for _, row in df.iterrows()
        ]
        
        # Verify first path as a sanity check before batch processing
        if not os.path.exists(xyz_paths[0]):
            raise FileNotFoundError(f"Missing XYZ geometry file: {xyz_paths[0]}")
        
        # Multi-core processing for efficiency (36 cores available)
        with ProcessPoolExecutor(max_workers=32) as executor:
            parsed_geometries = list(executor.map(parse_xyz_file, xyz_paths))
        
        # Map structural data back to the dataframe
        df['lattice_matrix'] = [item['lattice'] for item in parsed_geometries]
        df['atom_coords'] = [item['coords'] for item in parsed_geometries]
        df['atom_species'] = [item['species'] for item in parsed_geometries]
        return df

    # Data preparation (parsing XYZ)
    train_df = enrich_with_structural_data(train_df, "train")
    test_df = enrich_with_structural_data(test_df, "test")
    
    # Separate features and targets
    y_train = train_df[target_cols].copy()
    X_train = train_df.drop(columns=target_cols)
    X_test = test_df.copy()
    test_ids = test_df['id'].copy()
    
    # Row alignment checks
    if len(X_train) != len(y_train):
        raise ValueError(f"Training data alignment error: {len(X_train)} features vs {len(y_train)} targets.")
    if len(X_test) != len(test_ids):
        raise ValueError(f"Test data alignment error: {len(X_test)} features vs {len(test_ids)} IDs.")
        
    print(f"Data load successful.")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, test_ids
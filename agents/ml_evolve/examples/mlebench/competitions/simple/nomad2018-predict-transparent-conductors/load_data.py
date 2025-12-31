# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/2ea9a3e6-0185-40d8-bd93-7afe244a50a1/2/executor/output"

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
    # Load CSV files
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    def parse_geometry_file(file_path):
        """Parse a geometry.xyz file and extract lattice vectors and atom positions."""
        lattice_vectors = []
        atoms = {'Al': [], 'Ga': [], 'In': [], 'O': []}

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('lattice_vector'):
                    parts = line.split()
                    # lattice_vector x y z
                    vec = [float(parts[1]), float(parts[2]), float(parts[3])]
                    lattice_vectors.append(vec)
                elif line.startswith('atom'):
                    parts = line.split()
                    # atom x y z element
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    element = parts[4]
                    if element in atoms:
                        atoms[element].append([x, y, z])

        return lattice_vectors, atoms

    def extract_geometry_features(df, data_type='train'):
        """Extract features from geometry files for all samples in dataframe."""
        geometry_features = []

        for idx, row in df.iterrows():
            sample_id = int(row['id'])  # Ensure ID is an integer
            geometry_path = os.path.join(BASE_DATA_PATH, data_type, str(sample_id), "geometry.xyz")

            if not os.path.exists(geometry_path):
                raise FileNotFoundError(f"Geometry file not found: {geometry_path}")

            lattice_vectors, atoms = parse_geometry_file(geometry_path)

            features = {}

            # Lattice vector features
            lattice_vectors = np.array(lattice_vectors)
            if len(lattice_vectors) == 3:
                # Compute lattice vector magnitudes
                for i in range(3):
                    features[f'lv{i + 1}_magnitude'] = np.linalg.norm(lattice_vectors[i])

                # Compute volume using scalar triple product
                volume = np.abs(np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])))
                features['cell_volume'] = volume

                # Compute angles between lattice vectors
                def angle_between(v1, v2):
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    return np.degrees(np.arccos(cos_angle))

                features['lv_angle_12'] = angle_between(lattice_vectors[0], lattice_vectors[1])
                features['lv_angle_13'] = angle_between(lattice_vectors[0], lattice_vectors[2])
                features['lv_angle_23'] = angle_between(lattice_vectors[1], lattice_vectors[2])

            # Atom count features
            for element in ['Al', 'Ga', 'In', 'O']:
                features[f'n_{element}'] = len(atoms[element])

            # Total atoms from geometry
            features['total_atoms_geom'] = sum(len(atoms[e]) for e in atoms)

            # Atom position statistics for each element
            for element in ['Al', 'Ga', 'In', 'O']:
                if len(atoms[element]) > 0:
                    positions = np.array(atoms[element])
                    # Mean position
                    features[f'{element}_mean_x'] = np.mean(positions[:, 0])
                    features[f'{element}_mean_y'] = np.mean(positions[:, 1])
                    features[f'{element}_mean_z'] = np.mean(positions[:, 2])
                    # Std of positions
                    features[f'{element}_std_x'] = np.std(positions[:, 0])
                    features[f'{element}_std_y'] = np.std(positions[:, 1])
                    features[f'{element}_std_z'] = np.std(positions[:, 2])
                    # Distance from origin statistics
                    distances = np.linalg.norm(positions, axis=1)
                    features[f'{element}_dist_mean'] = np.mean(distances)
                    features[f'{element}_dist_std'] = np.std(distances)
                    features[f'{element}_dist_min'] = np.min(distances)
                    features[f'{element}_dist_max'] = np.max(distances)
                else:
                    features[f'{element}_mean_x'] = 0
                    features[f'{element}_mean_y'] = 0
                    features[f'{element}_mean_z'] = 0
                    features[f'{element}_std_x'] = 0
                    features[f'{element}_std_y'] = 0
                    features[f'{element}_std_z'] = 0
                    features[f'{element}_dist_mean'] = 0
                    features[f'{element}_dist_std'] = 0
                    features[f'{element}_dist_min'] = 0
                    features[f'{element}_dist_max'] = 0

            geometry_features.append(features)

        return pd.DataFrame(geometry_features)

    # Extract geometry features for train and test
    train_geom_features = extract_geometry_features(train_df, 'train')
    test_geom_features = extract_geometry_features(test_df, 'test')

    # Merge geometry features with CSV data
    train_merged = pd.concat([train_df.reset_index(drop=True), train_geom_features.reset_index(drop=True)], axis=1)
    test_merged = pd.concat([test_df.reset_index(drop=True), test_geom_features.reset_index(drop=True)], axis=1)

    # Define target columns
    target_columns = ['formation_energy_ev_natom', 'bandgap_energy_ev']

    # Extract targets
    y = train_merged[target_columns]

    # Define feature columns (exclude id and targets)
    feature_columns = [col for col in train_merged.columns if col not in ['id'] + target_columns]

    # Extract features
    X = train_merged[feature_columns]
    X_test = test_merged[feature_columns]

    # Extract test ids
    test_ids = test_merged['id']

    return X, y, X_test, test_ids

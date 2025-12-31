# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/832d0196-b83e-4fa9-8ea2-3588ff903a43/1/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def create_features(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT, DT, DT, DT]:
    """
    Creates GCN-ready graph features for RNA sequences.
    
    Args:
        X_train (DT): The training set features.
        y_train (DT): The training set labels.
        X_val (DT): The validation set features.
        y_val (DT): The validation set labels.
        X_test (DT): The test set features.
    
    Returns:
        Tuple[DT, DT, DT, DT, DT]: Transformed features and labels.
    """

    def get_adj_matrix(structure: str) -> np.ndarray:
        """Constructs and normalizes the adjacency matrix for the RNA graph."""
        seq_len = len(structure)
        # Initialize adjacency matrix A
        adj = np.zeros((seq_len, seq_len), dtype=np.float32)

        # 1. Consecutive bases (chemical bonds)
        for i in range(seq_len - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0

        # 2. Physical pairings (matching parentheses)
        stack = []
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # 3. Normalization: A_tilde = D^-1/2 * (A + I) * D^-1/2
        A_prime = adj + np.eye(seq_len, dtype=np.float32)
        degree = np.sum(A_prime, axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        # No division by zero possible as Dii >= 1 (self-loops)
        adj_norm = A_prime * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :]
        return adj_norm

    def get_node_features(sequence: str, structure: str, loop_type: str) -> np.ndarray:
        """One-hot encodes RNA sequence, structure, and loop type into a node feature matrix."""
        seq_len = len(sequence)
        # 4 (seq) + 3 (struct) + 7 (loop) = 14 features
        feats = np.zeros((seq_len, 14), dtype=np.float32)

        # Encoding maps based on competition specifications
        s_map = {'A': 0, 'G': 1, 'U': 2, 'C': 3}
        st_map = {'.': 0, '(': 1, ')': 2}
        l_map = {'S': 0, 'M': 1, 'I': 2, 'B': 3, 'H': 4, 'E': 5, 'X': 6}

        for i in range(seq_len):
            if sequence[i] in s_map:
                feats[i, s_map[sequence[i]]] = 1.0
            if structure[i] in st_map:
                feats[i, 4 + st_map[structure[i]]] = 1.0
            if loop_type[i] in l_map:
                feats[i, 7 + l_map[loop_type[i]]] = 1.0
        return feats

    def transform_dataset(df: DT) -> DT:
        """Applies graph transformations to each row in the dataset."""
        transformed_rows = []
        for i in range(len(df)):
            row = df.iloc[i]
            seq = row['sequence']
            struct = row['structure']
            loop = row['predicted_loop_type']

            nf = get_node_features(seq, struct, loop)
            adj = get_adj_matrix(struct)

            transformed_rows.append({
                'node_features': nf,
                'adj_matrix': adj,
                'id': row['id'],
                'seq_scored': row['seq_scored'],
                'seq_length': row['seq_length'],
                'signal_to_noise': row['signal_to_noise'],
                'SN_filter': row['SN_filter']
            })

        return pd.DataFrame(transformed_rows)

    # Step 1 & 2: Apply consistently to all sets (logic is rule-based, no "fitting" required)
    X_train_transformed = transform_dataset(X_train)
    X_val_transformed = transform_dataset(X_val)
    X_test_transformed = transform_dataset(X_test)

    # Target values are usually left unchanged but copied for safety
    y_train_transformed = y_train.copy()
    y_val_transformed = y_val.copy()

    # Step 3: Validation (ensure no NaNs/Infs and consistent columns)
    for dataset in [X_train_transformed, X_val_transformed, X_test_transformed]:
        if dataset.isna().any().any():
            raise ValueError("Transformed features contain NaN values.")

    return X_train_transformed, y_train_transformed, X_val_transformed, y_val_transformed, X_test_transformed

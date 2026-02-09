import numpy as np
import pandas as pd
from typing import Tuple, Any

# Task-adaptive type definitions
# X is a NumPy array of shape (N, 107, 111) combining node features and adjacency info
# y is a NumPy array of shape (N, 107, 5) containing target values
X_type = np.ndarray
y_type = np.ndarray

def preprocess(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[X_type, y_type, X_type, y_type, X_type]:
    """
    Transforms raw RNA data into model-ready format.
    
    Returns:
        - X_train_p: (N_train, 107, 111)
        - y_train_p: (N_train, 107, 5)
        - X_val_p: (N_val, 107, 111)
        - y_val_p: (N_val, 107, 5)
        - X_test_p: (N_test, 107, 111)
    """
    print("Starting preprocessing...")

    # Step 1: Align targets with features
    # The pipeline may pass subsets of X but full sets of y. We align using indices.
    def align(features, targets):
        if features is not None and targets is not None:
            if len(features) != len(targets):
                print(f"Aligning targets ({len(targets)}) to features ({len(features)}) using indices.")
                return targets.loc[features.index]
        return targets

    y_train_aligned = align(X_train, y_train)
    y_val_aligned = align(X_val, y_val)

    # Step 2: Configuration
    MAX_LEN = 107
    TARGET_COLS = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    
    # Mappings as per Technical Specification
    token_maps = {
        'sequence': {c: i for i, c in enumerate('ACGU')},
        'structure': {c: i for i, c in enumerate('.()')},
        'loop': {c: i for i, c in enumerate('BEHIMSX')}
    }

    def process_data(df_feat: pd.DataFrame, df_tgt: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """Processes features and targets into aligned arrays."""
        n_samples = len(df_feat)
        
        # Node features (N, 107, 4): seq, struct, loop, pos
        node_feats = np.zeros((n_samples, MAX_LEN, 4), dtype=np.float32)
        # Adjacency (N, 107, 107): backbone + base-pairing
        adj = np.zeros((n_samples, MAX_LEN, MAX_LEN), dtype=np.float32)
        # Targets (N, 107, 5)
        targets = np.zeros((n_samples, MAX_LEN, 5), dtype=np.float32)
        
        # Optimize by getting values upfront
        seqs = df_feat['sequence'].values
        structs = df_feat['structure'].values
        loops = df_feat['predicted_loop_type'].values
        
        for i in range(n_samples):
            seq = seqs[i]
            struct = structs[i]
            loop = loops[i]
            seq_len = len(seq)
            
            # 1. Node Features & Backbone Adjacency
            for j in range(min(seq_len, MAX_LEN)):
                node_feats[i, j, 0] = token_maps['sequence'].get(seq[j], 0)
                node_feats[i, j, 1] = token_maps['structure'].get(struct[j], 0)
                node_feats[i, j, 2] = token_maps['loop'].get(loop[j], 0)
                node_feats[i, j, 3] = j / MAX_LEN # Normalized Positional Encoding
                
                # Backbone connections
                if j < seq_len - 1 and j < MAX_LEN - 1:
                    adj[i, j, j+1] = 1.0
                    adj[i, j+1, j] = 1.0
            
            # 2. Base-pairing Adjacency
            stack = []
            for j, char in enumerate(struct):
                if j >= MAX_LEN: break
                if char == '(':
                    stack.append(j)
                elif char == ')':
                    if stack:
                        partner = stack.pop()
                        adj[i, j, partner] = 1.0
                        adj[i, partner, j] = 1.0
            
            # 3. Target Vectorization
            if df_tgt is not None:
                for t_idx, col in enumerate(TARGET_COLS):
                    if col in df_tgt.columns:
                        v = df_tgt[col].values[i]
                        l = min(len(v), MAX_LEN)
                        targets[i, :l, t_idx] = v[:l]
        
        # Combine node features and adjacency into a single X tensor to satisfy len(X) == n_samples
        # Concatenate on the feature dimension: (N, 107, 4 + 107) = (N, 107, 111)
        X_out = np.concatenate([node_feats, adj], axis=2)
        return X_out, targets

    # Step 3: Execute transformations
    print("Transforming training data...")
    X_train_p, y_train_p = process_data(X_train, y_train_aligned)
    
    print("Transforming validation data...")
    X_val_p, y_val_p = process_data(X_val, y_val_aligned)
    
    print("Transforming test data...")
    X_test_p, _ = process_data(X_test, None)

    # Step 4: Final Validation
    if len(X_train_p) != len(y_train_p):
        raise ValueError(f"Alignment error: X_train_p({len(X_train_p)}) != y_train_p({len(y_train_p)})")
    
    print(f"Preprocessing complete. Final shapes: Train X {X_train_p.shape}, y {y_train_p.shape}")
    
    return X_train_p, y_train_p, X_val_p, y_val_p, X_test_p
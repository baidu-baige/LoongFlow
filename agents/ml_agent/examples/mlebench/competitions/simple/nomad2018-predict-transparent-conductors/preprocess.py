import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.DataFrame

@torch.no_grad()
def compute_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates global and local structural features using GPU acceleration (PyTorch).
    Features include Unit Cell Volume, Density, Packing Fraction, and M-O distance stats.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Pre-allocate containers
    volumes = []
    densities = []
    packing_fractions = []
    mo_stats = {
        'Al': {'min': [], 'max': [], 'mean': [], 'std': []},
        'Ga': {'min': [], 'max': [], 'mean': [], 'std': []},
        'In': {'min': [], 'max': [], 'mean': [], 'std': []}
    }
    
    # Physical Constants
    # Atomic weights (amu)
    weights = {'Al': 26.9815, 'Ga': 69.723, 'In': 114.818, 'O': 15.999}
    # Empirical atomic radii (Angstrom)
    radii = {'Al': 1.43, 'Ga': 1.35, 'In': 1.67, 'O': 0.60}
    # Pre-calculate atomic volumes (4/3 * pi * r^3)
    atom_vols = {k: (4/3) * np.pi * (v**3) for k, v in radii.items()}
    
    for _, row in df.iterrows():
        # Load structure to GPU
        L = torch.as_tensor(row['lattice_matrix'], dtype=torch.float32, device=device)
        coords = torch.as_tensor(row['atom_coords'], dtype=torch.float32, device=device)
        species = row['atom_species']
        species_arr = np.array(species)
        
        # 1. Global Feature: Volume
        vol = torch.abs(torch.det(L)).item()
        volumes.append(vol)
        
        # 2. Global Features: Density & Packing Fraction
        unique, counts_arr = np.unique(species_arr, return_counts=True)
        counts = dict(zip(unique, counts_arr))
        
        total_mass = sum(counts.get(s, 0) * weights.get(s, 0) for s in weights)
        total_atom_vol = sum(counts.get(s, 0) * atom_vols.get(s, 0) for s in atom_vols)
        
        densities.append(total_mass / vol if vol > 0 else 0)
        packing_fractions.append(total_atom_vol / vol if vol > 0 else 0)
        
        # 3. Local Features: M-O Distance Statistics (PBC)
        inv_L = torch.inverse(L)
        S = coords @ inv_L # Fractional coordinates
        
        idx_O = np.where(species_arr == 'O')[0]
        s_O = S[idx_O]
        
        for m_type in ['Al', 'Ga', 'In']:
            idx_M = np.where(species_arr == m_type)[0]
            if len(idx_M) == 0 or len(idx_O) == 0:
                for stat in ['min', 'max', 'mean', 'std']:
                    mo_stats[m_type][stat].append(0.0)
                continue
            
            s_M = S[idx_M]
            
            # Minimum Image Convention PBC
            ds = s_O.unsqueeze(0) - s_M.unsqueeze(1)
            ds = ds - torch.round(ds)
            dx = ds @ L # Back to Cartesian
            dist = torch.norm(dx, dim=-1)
            
            # Apply cutoff for M-O bond identification
            dist_filtered = dist[dist < 4.0]
            
            if dist_filtered.numel() == 0:
                for stat in ['min', 'max', 'mean', 'std']:
                    mo_stats[m_type][stat].append(0.0)
            else:
                mo_stats[m_type]['min'].append(torch.min(dist_filtered).item())
                mo_stats[m_type]['max'].append(torch.max(dist_filtered).item())
                mo_stats[m_type]['mean'].append(torch.mean(dist_filtered).item())
                std_val = torch.std(dist_filtered).item() if dist_filtered.numel() > 1 else 0.0
                mo_stats[m_type]['std'].append(std_val)
                    
    struct_df = pd.DataFrame({
        'volume': volumes,
        'density': densities,
        'packing_fraction': packing_fractions
    }, index=df.index)
    
    for m_type in ['Al', 'Ga', 'In']:
        for stat in ['min', 'max', 'mean', 'std']:
            struct_df[f'{m_type}_O_{stat}'] = mo_stats[m_type][stat]
            
    return struct_df

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw crystal structural data and tabular features into a model-ready format.
    """
    print("Initiating preprocessing pipeline...")
    
    # Identify correct column names (handling potential truncation in EDA reports)
    all_cols = X_train.columns.tolist()
    def find_col(prefix):
        matches = [c for c in all_cols if c.startswith(prefix)]
        if not matches:
            raise KeyError(f"Column with prefix '{prefix}' not found in {all_cols}")
        return matches[0]

    num_total_col = find_col('number_of_total')
    lv1 = find_col('lattice_vector_1')
    lv2 = find_col('lattice_vector_2')
    lv3 = find_col('lattice_vector_3')
    la_alpha = find_col('lattice_angle_alpha')
    la_beta = find_col('lattice_angle_beta')
    la_gamma = find_col('lattice_angle_gamma')
    
    # 1. Fit One-Hot Encoder
    print("Fitting One-Hot Encoder for symmetry features...")
    # Using compatibility check for sparse parameter name
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe.fit(X_train[['spacegroup']])
    
    def transform_subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
        print(f"Extracting features for {name} subset...")
        
        # A. Structural Features (Global & Local)
        struct_feats = compute_structural_features(df)
        
        # B. Lattice Features: Log-transform skewed lattice vectors
        lat_logs = np.log1p(df[[lv1, lv2, lv3]])
        lat_logs.columns = [f"{lv1}_log", f"{lv2}_log", f"{lv3}_log"]
        
        # C. Symmetry Features: One-hot encode spacegroup
        ohe_cols = ohe.get_feature_names_out(['spacegroup'])
        ohe_df = pd.DataFrame(ohe.transform(df[['spacegroup']]), columns=ohe_cols, index=df.index)
        
        # D. Compositional & Static Features
        tabular_cols = [
            num_total_col, 
            'percent_atom_al', 'percent_atom_ga', 'percent_atom_in',
            la_alpha, la_beta, la_gamma
        ]
        tabular_feats = df[tabular_cols].copy()
        
        # E. Concatenate all features
        X_processed = pd.concat([tabular_feats, lat_logs, struct_feats, ohe_df], axis=1)
        
        # F. Post-processing
        X_processed = X_processed.replace([np.inf, -np.inf], 0).fillna(0)
        
        return X_processed

    # Apply transformations
    X_train_processed = transform_subset(X_train, "train")
    X_val_processed = transform_subset(X_val, "validation")
    X_test_processed = transform_subset(X_test, "test")
    
    y_train_processed = y_train.copy()
    y_val_processed = y_val.copy()
    
    print(f"Preprocessing complete. Feature count: {X_train_processed.shape[1]}")
    
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed
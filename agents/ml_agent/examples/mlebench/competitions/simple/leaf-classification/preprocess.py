import os
import cudf
import cuml
import numpy as np
import pandas as pd
from typing import Tuple, Any
from functools import partial
from cuml.preprocessing import RobustScaler, StandardScaler
from cuml.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA

# Task-adaptive type definitions using RAPIDS for GPU-accelerated processing
X = cudf.DataFrame      # Feature matrix type: RAPIDS DataFrame
y = cudf.Series         # Target vector type: RAPIDS Series

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw leaf data into a taxonomic-anchored, outlier-resistant discriminative manifold.
    
    Pipeline: RobustScaler -> PowerTransformer(Yeo-Johnson) -> SelectKBest(MI, k=200) -> 
              StandardScaler -> FeatureUnion(GroupedStats, PCA(50), Species-LDA(35), Genus-LDA(15), Cosine KPCA(20))
    """
    print("Preprocess Stage: Initializing Hybrid Taxonomic Manifold Pipeline.")
    
    # Set cuml to return cudf objects for GPU efficiency
    cuml.set_global_output_type('cudf')

    # Ensure float32 for consistency across GPU operations
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    # Extract Genus labels for Genus-LDA before potential transformation/dropping
    # Note: 'genus_feat' was added during load_data
    y_train_genus = X_train['genus_feat'].astype('int32')

    # 1. RobustScaler
    # Objective: Outlier-resistant foundation.
    print("Preprocess Stage: Applying RobustScaler...")
    rs = RobustScaler()
    X_train_rs = rs.fit_transform(X_train)
    X_val_rs = rs.transform(X_val)
    X_test_rs = rs.transform(X_test)
    
    X_train_rs.columns = X_train.columns
    X_val_rs.columns = X_val.columns
    X_test_rs.columns = X_test.columns

    # 2. PowerTransformer (Yeo-Johnson)
    # Objective: Minimize skewness across features.
    print("Preprocess Stage: Applying PowerTransformer (Yeo-Johnson)...")
    pt = PowerTransformer(method='yeo-johnson')
    
    # PT requires CPU/Numpy context
    X_train_pd = X_train_rs.to_pandas()
    X_train_pt_np = pt.fit_transform(X_train_pd)
    X_val_pt_np = pt.transform(X_val_rs.to_pandas())
    X_test_pt_np = pt.transform(X_test_rs.to_pandas())
    
    X_train_pt = pd.DataFrame(X_train_pt_np, index=X_train_pd.index, columns=X_train_pd.columns)
    X_val_pt = pd.DataFrame(X_val_pt_np, index=X_val_rs.to_pandas().index, columns=X_val_rs.columns)
    X_test_pt = pd.DataFrame(X_test_pt_np, index=X_test_rs.to_pandas().index, columns=X_test_rs.columns)

    # 3. SelectKBest (Mutual Information)
    # Objective: Filter non-discriminative noise.
    print("Preprocess Stage: Selecting top 200 features using Mutual Information...")
    k_features = min(200, X_train_pt.shape[1])
    score_func = partial(mutual_info_classif, random_state=42)
    skb = SelectKBest(score_func=score_func, k=k_features)
    
    y_train_pd = y_train.to_pandas()
    X_train_selected_np = skb.fit_transform(X_train_pt, y_train_pd)
    X_val_selected_np = skb.transform(X_val_pt)
    X_test_selected_np = skb.transform(X_test_pt)
    
    selected_mask = skb.get_support()
    selected_cols = X_train_pt.columns[selected_mask]
    
    X_train_sel = cudf.DataFrame(X_train_selected_np, index=X_train.index, columns=selected_cols).astype('float32')
    X_val_sel = cudf.DataFrame(X_val_selected_np, index=X_val.index, columns=selected_cols).astype('float32')
    X_test_sel = cudf.DataFrame(X_test_selected_np, index=X_test.index, columns=selected_cols).astype('float32')

    # 4. StandardScaler
    # Objective: Normalization for projection components.
    print("Preprocess Stage: Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    X_train_scaled.columns = selected_cols
    X_val_scaled.columns = selected_cols
    X_test_scaled.columns = selected_cols

    # 5. FeatureUnion Components
    
    # Component A: Grouped Stats
    def extract_grouped_stats(df: cudf.DataFrame) -> cudf.DataFrame:
        stats_data = {}
        all_cols = df.columns.tolist()
        groups = {
            'margin': [c for c in all_cols if 'margin' in str(c)],
            'shape': [c for c in all_cols if 'shape' in str(c)],
            'texture': [c for c in all_cols if 'texture' in str(c)]
        }
        for name, cols in groups.items():
            if len(cols) > 0:
                subset = df[cols]
                stats_data[f'{name}_mean'] = subset.mean(axis=1)
                stats_data[f'{name}_std'] = subset.std(axis=1)
                stats_data[f'{name}_max'] = subset.max(axis=1)
            else:
                stats_data[f'{name}_mean'] = cudf.Series(0.0, index=df.index)
                stats_data[f'{name}_std'] = cudf.Series(0.0, index=df.index)
                stats_data[f'{name}_max'] = cudf.Series(0.0, index=df.index)
        return cudf.DataFrame(stats_data, index=df.index).astype('float32')

    X_train_stats = extract_grouped_stats(X_train_scaled)
    X_val_stats = extract_grouped_stats(X_val_scaled)
    X_test_stats = extract_grouped_stats(X_test_scaled)

    # Component B: Global PCA (50 components)
    print("Preprocess Stage: Projecting Global PCA (50 components)...")
    pca_n = min(50, X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=pca_n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    pca_cols = [f'pca_{i}' for i in range(pca_n)]
    X_train_pca.columns = pca_cols
    X_val_pca.columns = pca_cols
    X_test_pca.columns = pca_cols

    # Component C: Species-LDA (35 components)
    print("Preprocess Stage: Computing Species-LDA (35 components)...")
    n_classes_s = int(y_train.nunique())
    slda_n = min(35, n_classes_s - 1, X_train_scaled.shape[1])
    slda = LinearDiscriminantAnalysis(n_components=slda_n, shrinkage='auto', solver='eigen')
    
    X_train_scaled_pd = X_train_scaled.to_pandas()
    X_train_slda_np = slda.fit_transform(X_train_scaled_pd, y_train_pd)
    X_val_slda_np = slda.transform(X_val_scaled.to_pandas())
    X_test_slda_np = slda.transform(X_test_scaled.to_pandas())
    
    slda_cols = [f'slda_{i}' for i in range(slda_n)]
    X_train_slda = cudf.DataFrame(X_train_slda_np, index=X_train.index, columns=slda_cols).astype('float32')
    X_val_slda = cudf.DataFrame(X_val_slda_np, index=X_val.index, columns=slda_cols).astype('float32')
    X_test_slda = cudf.DataFrame(X_test_slda_np, index=X_test.index, columns=slda_cols).astype('float32')

    # Component D: Genus-LDA (15 components)
    print("Preprocess Stage: Computing Genus-LDA (15 components)...")
    y_train_genus_pd = y_train_genus.to_pandas()
    n_classes_g = int(y_train_genus.nunique())
    glda_n = min(15, n_classes_g - 1, X_train_scaled.shape[1])
    glda = LinearDiscriminantAnalysis(n_components=glda_n, shrinkage='auto', solver='eigen')
    
    X_train_glda_np = glda.fit_transform(X_train_scaled_pd, y_train_genus_pd)
    X_val_glda_np = glda.transform(X_val_scaled.to_pandas())
    X_test_glda_np = glda.transform(X_test_scaled.to_pandas())
    
    glda_cols = [f'glda_{i}' for i in range(glda_n)]
    X_train_glda = cudf.DataFrame(X_train_glda_np, index=X_train.index, columns=glda_cols).astype('float32')
    X_val_glda = cudf.DataFrame(X_val_glda_np, index=X_val.index, columns=glda_cols).astype('float32')
    X_test_glda = cudf.DataFrame(X_test_glda_np, index=X_test.index, columns=glda_cols).astype('float32')

    # Component E: Cosine KernelPCA (20 components)
    print("Preprocess Stage: Computing Cosine KernelPCA (20 components)...")
    kpca_n = min(20, X_train_scaled.shape[0], X_train_scaled.shape[1])
    kpca = KernelPCA(n_components=kpca_n, kernel='cosine')
    
    X_train_kpca_np = kpca.fit_transform(X_train_scaled_pd)
    X_val_kpca_np = kpca.transform(X_val_scaled.to_pandas())
    X_test_kpca_np = kpca.transform(X_test_scaled.to_pandas())
    
    kpca_cols = [f'kpca_{i}' for i in range(kpca_n)]
    X_train_kpca = cudf.DataFrame(X_train_kpca_np, index=X_train.index, columns=kpca_cols).astype('float32')
    X_val_kpca = cudf.DataFrame(X_val_kpca_np, index=X_val.index, columns=kpca_cols).astype('float32')
    X_test_kpca = cudf.DataFrame(X_test_kpca_np, index=X_test.index, columns=kpca_cols).astype('float32')

    # 6. Feature Fusion
    print("Preprocess Stage: Fusing manifold components.")
    X_train_processed = cudf.concat([X_train_stats, X_train_pca, X_train_slda, X_train_glda, X_train_kpca], axis=1)
    X_val_processed = cudf.concat([X_val_stats, X_val_pca, X_val_slda, X_val_glda, X_val_kpca], axis=1)
    X_test_processed = cudf.concat([X_test_stats, X_test_pca, X_test_slda, X_test_glda, X_test_kpca], axis=1)

    # 7. Integrity Verification
    def clean_manifold(df: cudf.DataFrame) -> cudf.DataFrame:
        if df.isna().any().any():
            df = df.fillna(0.0)
        is_inf = (df == np.inf) | (df == -np.inf)
        if is_inf.any().any():
            df = df.replace([np.inf, -np.inf], 0.0)
        return df

    X_train_processed = clean_manifold(X_train_processed)
    X_val_processed = clean_manifold(X_val_processed)
    X_test_processed = clean_manifold(X_test_processed)

    print(f"Preprocess Stage Complete. Manifold dimensions: {X_train_processed.shape[1]}")
    
    return (
        X_train_processed, 
        y_train, 
        X_val_processed, 
        y_val, 
        X_test_processed
    )
# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import numpy as np
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/nomad2018-predict-transparent-conductors/prepared/public"
OUTPUT_DATA_PATH = "output/2ea9a3e6-0185-40d8-bd93-7afe244a50a1/2/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def create_features(
    X_train: DT,
    y_train: DT,
    X_test: DT
) -> Tuple[DT, DT, DT]:
    """
    Creates features for a single fold of cross-validation.
    """

    def compute_composition_features(df):
        """Compute composition-based features from CSV data."""
        features = pd.DataFrame(index=df.index)

        # Get composition percentages
        al = df['percent_atom_al'].values
        ga = df['percent_atom_ga'].values
        in_ = df['percent_atom_in'].values

        # Interaction terms
        features['al_ga'] = al * ga
        features['al_in'] = al * in_
        features['ga_in'] = ga * in_
        features['al_ga_in'] = al * ga * in_

        # Squared terms
        features['al_sq'] = al ** 2
        features['ga_sq'] = ga ** 2
        features['in_sq'] = in_ ** 2

        # Ratios (with small epsilon to avoid division by zero)
        eps = 1e-8
        features['al_to_ga'] = al / (ga + eps)
        features['al_to_in'] = al / (in_ + eps)
        features['ga_to_in'] = ga / (in_ + eps)
        features['ga_to_al'] = ga / (al + eps)
        features['in_to_al'] = in_ / (al + eps)
        features['in_to_ga'] = in_ / (ga + eps)

        # Entropy of composition
        def compute_entropy(row):
            percentages = [row['percent_atom_al'], row['percent_atom_ga'], row['percent_atom_in']]
            entropy = 0
            for p in percentages:
                if p > 0:
                    entropy -= p * np.log(p + eps)
            return entropy

        features['composition_entropy'] = df.apply(compute_entropy, axis=1)

        return features

    def compute_lattice_features(df):
        """Compute lattice-based features from CSV data."""
        features = pd.DataFrame(index=df.index)

        lv1 = df['lattice_vector_1_ang'].values
        lv2 = df['lattice_vector_2_ang'].values
        lv3 = df['lattice_vector_3_ang'].values
        alpha = df['lattice_angle_alpha_degree'].values
        beta = df['lattice_angle_beta_degree'].values
        gamma = df['lattice_angle_gamma_degree'].values
        n_atoms = df['number_of_total_atoms'].values

        # Volume approximation
        features['volume_approx'] = lv1 * lv2 * lv3

        # Volume per atom
        features['volume_per_atom'] = features['volume_approx'] / n_atoms

        # Lattice vector ratios
        eps = 1e-8
        features['lv1_to_lv2'] = lv1 / (lv2 + eps)
        features['lv2_to_lv3'] = lv2 / (lv3 + eps)
        features['lv1_to_lv3'] = lv1 / (lv3 + eps)

        # Angle deviations from 90 degrees
        features['alpha_dev'] = np.abs(alpha - 90)
        features['beta_dev'] = np.abs(beta - 90)
        features['gamma_dev'] = np.abs(gamma - 90)

        # Sum of angle deviations
        features['total_angle_dev'] = features['alpha_dev'] + features['beta_dev'] + features['gamma_dev']

        # Additional lattice features
        features['lv_sum'] = lv1 + lv2 + lv3
        features['lv_mean'] = (lv1 + lv2 + lv3) / 3
        features['lv_std'] = np.std(np.column_stack([lv1, lv2, lv3]), axis=1)

        return features

    def one_hot_encode_spacegroup(df, all_spacegroups):
        """One-hot encode spacegroup column."""
        features = pd.DataFrame(index=df.index)

        for sg in all_spacegroups:
            features[f'spacegroup_{sg}'] = (df['spacegroup'] == sg).astype(int)

        return features

    # Get all unique spacegroups from training data
    all_spacegroups = sorted(X_train['spacegroup'].unique())

    # Process training data
    X_train_transformed = X_train.copy()

    # Add composition features
    comp_features_train = compute_composition_features(X_train)
    X_train_transformed = pd.concat([X_train_transformed.reset_index(drop=True),
                                     comp_features_train.reset_index(drop=True)], axis=1)

    # Add lattice features
    lattice_features_train = compute_lattice_features(X_train)
    X_train_transformed = pd.concat([X_train_transformed.reset_index(drop=True),
                                     lattice_features_train.reset_index(drop=True)], axis=1)

    # Add one-hot encoded spacegroup
    spacegroup_features_train = one_hot_encode_spacegroup(X_train, all_spacegroups)
    X_train_transformed = pd.concat([X_train_transformed.reset_index(drop=True),
                                     spacegroup_features_train.reset_index(drop=True)], axis=1)

    # Process test data
    X_test_transformed = X_test.copy()

    # Add composition features
    comp_features_test = compute_composition_features(X_test)
    X_test_transformed = pd.concat([X_test_transformed.reset_index(drop=True),
                                    comp_features_test.reset_index(drop=True)], axis=1)

    # Add lattice features
    lattice_features_test = compute_lattice_features(X_test)
    X_test_transformed = pd.concat([X_test_transformed.reset_index(drop=True),
                                    lattice_features_test.reset_index(drop=True)], axis=1)

    # Add one-hot encoded spacegroup
    spacegroup_features_test = one_hot_encode_spacegroup(X_test, all_spacegroups)
    X_test_transformed = pd.concat([X_test_transformed.reset_index(drop=True),
                                    spacegroup_features_test.reset_index(drop=True)], axis=1)

    # Remove original spacegroup column (now one-hot encoded)
    if 'spacegroup' in X_train_transformed.columns:
        X_train_transformed = X_train_transformed.drop(columns=['spacegroup'])
    if 'spacegroup' in X_test_transformed.columns:
        X_test_transformed = X_test_transformed.drop(columns=['spacegroup'])

    # Transform targets using log1p for RMSLE optimization
    y_train_transformed = y_train.copy()
    y_train_transformed = np.log1p(y_train_transformed)

    # Handle any NaN or Infinity values
    # Replace inf with large finite values
    X_train_transformed = X_train_transformed.replace([np.inf, -np.inf], np.nan)
    X_test_transformed = X_test_transformed.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with column means from training data
    train_means = X_train_transformed.mean()
    X_train_transformed = X_train_transformed.fillna(train_means)
    X_test_transformed = X_test_transformed.fillna(train_means)

    # Final check - fill any remaining NaN with 0
    X_train_transformed = X_train_transformed.fillna(0)
    X_test_transformed = X_test_transformed.fillna(0)

    # Ensure column consistency
    # Get common columns
    train_cols = set(X_train_transformed.columns)
    test_cols = set(X_test_transformed.columns)
    common_cols = list(train_cols.intersection(test_cols))

    # Reorder columns to be consistent
    X_train_transformed = X_train_transformed[common_cols]
    X_test_transformed = X_test_transformed[common_cols]

    return X_train_transformed, y_train_transformed, X_test_transformed

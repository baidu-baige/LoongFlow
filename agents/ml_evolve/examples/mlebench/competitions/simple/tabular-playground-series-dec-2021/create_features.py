# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Tuple

import cudf
import pandas as pd

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/tabular-playground-series-dec-2021/prepared/public"
OUTPUT_DATA_PATH = "output/ead961ce-50a1-41ec-89e9-4cc0d527fbe5/1/executor/output"

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
    Creates features for a single fold of cross-validation using GPU acceleration.
    """

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        # Move to GPU for accelerated processing
        gdf = cudf.from_pandas(df)

        # 1. Geometric Features
        # Euclidean distance: sqrt(H^2 + V^2)
        gdf['Euclidean_Distance_To_Hydrology'] = (gdf['Horizontal_Distance_To_Hydrology'] ** 2 + gdf[
            'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
        # Manhattan distance: |H| + |V|
        gdf['Manhattan_Distance_To_Hydrology'] = gdf['Horizontal_Distance_To_Hydrology'].abs() + gdf[
            'Vertical_Distance_To_Hydrology'].abs()

        # 2. Aspect Handling: Wrap Aspect values to [0, 360) using modulo
        gdf['Aspect_360'] = gdf['Aspect'] % 360

        # 3. Hillshade Aggregations
        hillshade_cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
        gdf['Hillshade_Mean'] = gdf[hillshade_cols].mean(axis=1)
        gdf['Hillshade_Range'] = gdf[hillshade_cols].max(axis=1) - gdf[hillshade_cols].min(axis=1)

        # 4. Soil Type & Wilderness Area Counts: Sum the binary indicator columns
        # Dynamically identify columns as some might have been dropped in load_data (e.g., constant columns)
        soil_cols = [c for c in gdf.columns if 'Soil_Type' in c]
        wild_cols = [c for c in gdf.columns if 'Wilderness_Area' in c]
        gdf['Soil_Type_Count'] = gdf[soil_cols].sum(axis=1)
        gdf['Wilderness_Area_Count'] = gdf[wild_cols].sum(axis=1)

        # 5. Elevation Adjustments
        gdf['Elevation_minus_Vertical_Hydrology'] = gdf['Elevation'] - gdf['Vertical_Distance_To_Hydrology']

        # 6. Memory Optimization & Numeric Stability
        # Convert all floats to float32 and ints to int32 (consistent with load_data)
        for col in gdf.columns:
            if gdf[col].dtype == 'float64':
                gdf[col] = gdf[col].astype('float32')
            elif gdf[col].dtype == 'int64':
                gdf[col] = gdf[col].astype('int32')

        # Clean up: ensure no NaNs/Infs (though none expected given the operations and clean input)
        # Note: Guidance says let errors propagate, so we avoid fallback filling.

        # Move back to pandas for the pipeline
        return gdf.to_pandas()

    # Apply the same transformation logic to training, validation, and test sets
    # This ensures column consistency and prevents data leakage (no fit-parameters used)
    X_train_transformed = transform(X_train)
    X_val_transformed = transform(X_val)
    X_test_transformed = transform(X_test)

    # Labels were already pre-processed (label encoded) in load_data and are returned unchanged
    y_train_transformed = y_train
    y_val_transformed = y_val

    return X_train_transformed, y_train_transformed, X_val_transformed, y_val_transformed, X_test_transformed

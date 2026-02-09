from typing import Tuple, Any
import pandas as pd
import numpy as np

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X, y, X, y, X]:
    """
    Transforms raw data into model-ready format for a single fold/split.
    Implements feature engineering to capture spatial relationships and reduces noise by 
    dropping constant columns.
    """
    
    def transform_features(df: pd.DataFrame) -> pd.DataFrame:
        """Applies the technical specification's feature engineering to a dataframe."""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        print("Creating spatial and aggregate features...")
        
        # Euclidean_Distance_To_Hydrology: sqrt(Horizontal**2 + Vertical**2)
        df['Euclidean_Distance_To_Hydrology'] = np.sqrt(
            df['Horizontal_Distance_To_Hydrology'].astype(np.float32)**2 + 
            df['Vertical_Distance_To_Hydrology'].astype(np.float32)**2
        ).astype(np.float32)
        
        # Manhattan_Distance_To_Hydrology: abs(Horizontal) + abs(Vertical)
        df['Manhattan_Distance_To_Hydrology'] = (
            df['Horizontal_Distance_To_Hydrology'].abs() + 
            df['Vertical_Distance_To_Hydrology'].abs()
        ).astype(np.float32)
        
        # Hillshade_Sum: Hillshade_9am + Hillshade_Noon + Hillshade_3pm
        df['Hillshade_Sum'] = (
            df['Hillshade_9am'].astype(np.int32) + 
            df['Hillshade_Noon'].astype(np.int32) + 
            df['Hillshade_3pm'].astype(np.int32)
        ).astype(np.int32)
        
        # Hillshade_Diff_93: Hillshade_9am - Hillshade_3pm
        df['Hillshade_Diff_93'] = (
            df['Hillshade_9am'].astype(np.int32) - 
            df['Hillshade_3pm'].astype(np.int32)
        ).astype(np.int32)
        
        # Aspect_Mod: Aspect % 360
        df['Aspect_Mod'] = (df['Aspect'] % 360).astype(np.int32)
        
        # Soil_Type_Count: Row-wise sum of all Soil_Type binary columns
        soil_cols = [col for col in df.columns if 'Soil_Type' in col]
        df['Soil_Type_Count'] = df[soil_cols].sum(axis=1).astype(np.int8)
        
        # Wilderness_Area_Count: Row-wise sum of all Wilderness_Area binary columns
        wild_cols = [col for col in df.columns if 'Wilderness_Area' in col]
        df['Wilderness_Area_Count'] = df[wild_cols].sum(axis=1).astype(np.int8)
        
        # Drop constant columns: Soil_Type7, Soil_Type15
        cols_to_drop = ['Soil_Type7', 'Soil_Type15']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        return df

    # Step 1 & 2: Apply transformations to train, val, and test sets consistently
    print("Pre-processing training data...")
    X_train_processed = transform_features(X_train)
    
    print("Pre-processing validation data...")
    X_val_processed = transform_features(X_val)
    
    print("Pre-processing test data...")
    X_test_processed = transform_features(X_test)
    
    # Targets remain unchanged as per specification
    y_train_processed = y_train
    y_val_processed = y_val

    # Step 3: Validate output format (no NaN/Inf, consistent structure)
    def validate_processed_data(df: pd.DataFrame, name: str):
        if df.isna().any().any():
            raise ValueError(f"NaN values detected in {name} after preprocessing.")
        if np.isinf(df.select_dtypes(include=np.number)).any().any():
            raise ValueError(f"Infinite values detected in {name} after preprocessing.")
            
    validate_processed_data(X_train_processed, "X_train")
    validate_processed_data(X_val_processed, "X_val")
    validate_processed_data(X_test_processed, "X_test")
    
    # Ensure column consistency
    if not (list(X_train_processed.columns) == list(X_val_processed.columns) == list(X_test_processed.columns)):
        raise ValueError("Processed feature sets have inconsistent column structures.")

    print(f"Preprocessing complete. Final feature count: {X_train_processed.shape[1]}")
    
    # Step 4: Return transformed data
    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed
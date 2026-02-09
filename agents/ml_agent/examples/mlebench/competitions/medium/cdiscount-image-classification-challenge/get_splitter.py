import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from typing import Any, Tuple, Iterator

# Task-adaptive type definitions
X = pd.DataFrame      # Feature metadata (offsets, lengths, product IDs)
y = pd.DataFrame      # Target indices (category, level1, level2)

class ProductStratifiedShuffleSplitter:
    """
    A splitter that performs stratified shuffle splitting at the product level.
    This ensures that all images of the same product stay within the same split,
    preventing data leakage, while maintaining the distribution of the Level 1 categories.
    
    In cases where stratification is impossible (e.g., small subsets where classes 
    have only 1 member), it falls back to a standard ShuffleSplit at the product level.
    """
    def __init__(self, test_size: float = 0.05, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = 1

    def split(self, X: pd.DataFrame, y: pd.DataFrame, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices for training and validation splits at the image level.
        """
        # Step 1: Create a lightweight metadata table for unique products
        # We need the product ID (_id) for grouping and the Level 1 category (l1_idx) for stratification.
        # Note: X and y are mapped 1-to-1 at the image level.
        product_metadata = pd.DataFrame({
            '_id': X['_id'].values,
            'l1_idx': y['l1_idx'].values
        }).drop_duplicates(subset=['_id'])
        
        # Step 2: Determine if Stratified split is possible
        # Stratification requires at least 2 samples per class to split into train/val.
        counts = product_metadata['l1_idx'].value_counts()
        if counts.min() < 2:
            # Fallback to standard ShuffleSplit if stratification is mathematically impossible
            # This often happens during 'validation_mode' with small subsets.
            splitter = ShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                random_state=self.random_state
            )
            split_iter = splitter.split(product_metadata)
        else:
            # Standard StratifiedShuffleSplit as per technical specification
            splitter = StratifiedShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                random_state=self.random_state
            )
            split_iter = splitter.split(product_metadata, product_metadata['l1_idx'])
        
        # Step 3: Map product-level split back to image-level indices
        for train_prod_idx, val_prod_idx in split_iter:
            # Get the set of product IDs for training
            train_pids = product_metadata.iloc[train_prod_idx]['_id'].values
            
            # Use isin() to find all images belonging to these products
            # This ensures absolute isolation: all images of a product are in one split.
            train_mask = X['_id'].isin(train_pids).values
            
            yield np.where(train_mask)[0], np.where(~train_mask)[0]

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.
    The strategy implements Product-Stratified Shuffle Split to prevent leakage
    and ensure categorical representation.
    """
    return ProductStratifiedShuffleSplitter(
        test_size=0.05, 
        random_state=42
    )
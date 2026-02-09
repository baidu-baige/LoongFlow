import numpy as np
from typing import List, Iterator, Tuple
from sklearn.model_selection import KFold

# Task-adaptive type definitions
X = List[np.ndarray]  # List of 2D grayscale images (H, W) as float32 in [0, 1]
y = List[np.ndarray]  # List of 2D grayscale images (H, W) as float32 in [0, 1]


def get_splitter(X: X, y: y) -> KFold:
    """
    Defines and returns a K-Fold cross-validation strategy for the denoising task.

    Given the small dataset size (115 images), a 5-fold cross-validation is used
    to provide a robust estimate of model performance and maximize training data usage.
    The splitting is performed at the image level to ensure that if images are later
    processed into patches, all patches from a single image remain within the same
    fold, preventing data leakage.

    Args:
        X (X): The full training features (list of dirty images).
        y (y): The full training targets (list of cleaned images).

    Returns:
        KFold: A scikit-learn KFold instance configured with 5 splits,
               shuffling enabled, and a fixed random seed for reproducibility.
    """
    # Technical Specification: K-Fold (K=5), shuffle=True, random_state=42.
    # This ensures that we evaluate on different subsets of the 115 images.
    splitter = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    return splitter
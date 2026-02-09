from typing import Any, Iterator, Tuple
import cudf
import cupy as cp

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/h-and-m-personalized-fashion-recommendations/prepared/public"
OUTPUT_DATA_PATH = "output/083259f0-3b2a-44c4-af6c-8557ef06ad6a/8/executor/output"

# Task-adaptive type definitions using GPU-accelerated containers
X = cudf.DataFrame
y = cudf.Series

class TemporalHoldoutSplitter:
    """
    Implements a temporal holdout splitting strategy using GPU-accelerated operations.
    This splitter isolates a specific time window for validation to mimic the 
    competition's 7-day test period evaluation based on the final week of training data.
    """
    def __init__(self, val_start_date: str, val_end_date: str):
        # Convert to datetime objects for efficient comparison with t_dat column
        self.val_start_date = cudf.to_datetime(val_start_date)
        self.val_end_date = cudf.to_datetime(val_end_date)

    def get_n_splits(self, X: X = None, y: y = None, groups: Any = None) -> int:
        """Returns the number of splitting iterations."""
        return 1

    def split(self, X: X, y: y = None, groups: Any = None) -> Iterator[Tuple[cp.ndarray, cp.ndarray]]:
        """
        Generates indices to split data into training and validation sets.
        
        Args:
            X (cudf.DataFrame): Feature matrix containing the 't_dat' column.
            y (cudf.Series, optional): Target vector.
            groups (Any, optional): Group labels.

        Yields:
            Tuple[cp.ndarray, cp.ndarray]: Tuple containing (train_indices, val_indices) 
                                           as CuPy arrays.
        """
        if 't_dat' not in X.columns:
            # Propagate error immediately if required column is missing
            raise KeyError("The feature matrix X must contain 't_dat' for temporal splitting strategy.")

        # Create boolean masks on the GPU
        # Validation set: the final 7 days (2020-09-08 to 2020-09-14)
        is_val = (X['t_dat'] >= self.val_start_date) & (X['t_dat'] <= self.val_end_date)
        # Training set: all transactions strictly before the validation period
        is_train = X['t_dat'] < self.val_start_date

        # Generate integer indices using CuPy for GPU memory efficiency
        indices = cp.arange(len(X), dtype=cp.int32)
        
        # Filter indices using the boolean masks
        # .values on a cudf.Series returns the underlying CuPy array
        train_idx = indices[is_train.values]
        val_idx = indices[is_val.values]

        yield train_idx, val_idx

def get_splitter(X: X, y: y) -> Any:
    """
    Defines and returns a data splitting strategy for model validation.

    This function determines HOW to partition data for training vs validation.
    The strategy implements a Temporal Holdout on the final 7 days (2020-09-08 to 2020-09-14)
    to align with the task's evaluation requirements.

    Args:
        X (X): The full training features (cuDF DataFrame).
        y (y): The full training targets (cuDF Series).

    Returns:
        TemporalHoldoutSplitter: A splitter object that isolates the final 7 days 
                                 for validation.
    """
    # Step 1: Analyze task type and data characteristics
    # The task is a recommendation problem with a temporal component. 
    # Validating on the last known week is the most realistic proxy for the unseen test set.

    # Step 2: Define specific date parameters derived from technical specification
    val_start_date = '2020-09-08'
    val_end_date = '2020-09-14'

    # Step 3: Return configured splitter instance
    # The splitter is reproducible as it is based on fixed dates.
    return TemporalHoldoutSplitter(
        val_start_date=val_start_date,
        val_end_date=val_end_date
    )
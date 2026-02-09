import os
import cudf
from typing import Tuple

# Task-adaptive type definitions using GPU-accelerated cuDF
X = cudf.DataFrame
y = cudf.Series
Ids = cudf.Series

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/h-and-m-personalized-fashion-recommendations/prepared/public"
OUTPUT_DATA_PATH = "output/083259f0-3b2a-44c4-af6c-8557ef06ad6a/8/executor/output"

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the H&M dataset using GPU acceleration via cuDF.
    
    Implementation details:
    - Strictly typed loading: article_id (int32), age (int8), customer_id (string).
    - GPU-accelerated processing with cuDF for memory efficiency and speed.
    - Aggressive downcasting and date filtering to handle 32GB raw data.
    - Captures specific metadata columns: product_type_name, colour_group_name, graphical_appearance_name, index_name, department_name.
    - Filters transactions from '2020-05-01' onwards.
    """
    print("Starting GPU-accelerated data loading...")

    # Define paths
    articles_path = os.path.join(BASE_DATA_PATH, "articles.csv")
    customers_path = os.path.join(BASE_DATA_PATH, "customers.csv")
    transactions_path = os.path.join(BASE_DATA_PATH, "transactions_train.csv")
    sample_sub_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")

    # Step 0: Ensure data existence
    for path in [articles_path, customers_path, transactions_path, sample_sub_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")

    # Step 1: Load Article metadata
    print("Loading articles metadata with requested columns...")
    article_cols = [
        'article_id', 'product_type_name', 'colour_group_name', 
        'graphical_appearance_name', 'index_name', 'department_name'
    ]
    # Read article_id as str first to handle leading zeros, then cast to int32 to match transactions
    articles = cudf.read_csv(articles_path, usecols=article_cols, dtype={'article_id': str})
    articles['article_id'] = articles['article_id'].astype('int32')

    # Step 2: Load Customer metadata
    print("Loading and cleaning customer metadata...")
    customer_cols = ['customer_id', 'age', 'FN', 'Active']
    customers = cudf.read_csv(customers_path, usecols=customer_cols, dtype={'customer_id': str})
    
    # Fill missing age with median and cast to int8
    age_median = customers['age'].median()
    customers['age'] = customers['age'].fillna(age_median).astype('int8')
    
    # Cast flags to int8 (assuming 1.0 is active, NaN is inactive)
    customers['FN'] = customers['FN'].fillna(0).astype('int8')
    customers['Active'] = customers['Active'].fillna(0).astype('int8')

    # Step 3: Load Transactions
    print("Loading and filtering transactions since 2020-05-01...")
    trans_dtype = {
        'customer_id': 'str',
        'article_id': 'int32',
        'price': 'float32',
        'sales_channel_id': 'int8'
    }
    # Load transactions and convert date
    transactions = cudf.read_csv(transactions_path, dtype=trans_dtype)
    transactions['t_dat'] = cudf.to_datetime(transactions['t_dat'])
    
    # Filter by start_date to optimize memory usage
    start_date = '2020-05-01'
    transactions = transactions[transactions['t_dat'] >= start_date].reset_index(drop=True)

    # Step 4: Data Preparation Persistence
    # Always prepare full data regardless of validation_mode for downstream consistency
    cleaned_dir = os.path.join(OUTPUT_DATA_PATH, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)
    articles.to_parquet(os.path.join(cleaned_dir, "articles_cleaned.parquet"))
    customers.to_parquet(os.path.join(cleaned_dir, "customers_cleaned.parquet"))
    transactions.to_parquet(os.path.join(cleaned_dir, "transactions_cleaned.parquet"))
    print(f"Full cleaned datasets saved to {cleaned_dir}")

    # Step 5: Structure Training Data (X_train, y_train)
    print("Merging transactions with metadata for training set...")
    # X_train combines transaction context with item and customer metadata
    X_train = transactions.merge(articles, on='article_id', how='left')
    X_train = X_train.merge(customers, on='customer_id', how='left')
    
    # y_train is the target article_id as specified
    y_train = X_train['article_id']

    # Step 6: Structure Test Data (X_test, test_ids)
    print("Preparing test set context from sample submission...")
    # We must make predictions for all customer_id values in sample_submission
    sample_sub = cudf.read_csv(sample_sub_path, usecols=['customer_id'], dtype={'customer_id': 'str'})
    test_ids = sample_sub['customer_id']
    X_test = sample_sub.merge(customers, on='customer_id', how='left')

    # Step 7: Handle validation mode subsetting
    if validation_mode:
        print("Validation mode enabled: Subsetting data to 200 rows.")
        X_train = X_train.head(200)
        y_train = y_train.head(200)
        X_test = X_test.head(200)
        test_ids = test_ids.head(200)

    # Final sanity checks
    assert len(X_train) == len(y_train), "Sample alignment mismatch in training set."
    assert len(X_test) == len(test_ids), "Sample alignment mismatch in test set."
    assert not X_train.empty, "Training feature matrix is empty."
    assert not X_test.empty, "Test feature matrix is empty."

    print(f"Data loading complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, y_train, X_test, test_ids
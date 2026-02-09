import pandas as pd
import os
import numpy as np
from typing import Tuple
from datasets import load_dataset

# Target paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/chaii-hindi-and-tamil-question-answering/prepared/public"
OUTPUT_DATA_PATH = "output/af0f7d71-a062-46e3-8926-51aedd28d3b4/3/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame  # Feature matrix: id, context, question, language
y = pd.DataFrame  # Target vector: answer_text, answer_start
Ids = pd.Series   # Identifier type: id

def load_data(validation_mode: bool = False) -> Tuple[X, y, X, Ids]:
    """
    Loads and prepares the datasets required for the Hindi/Tamil QA task.
    Augments the competition training data with TyDiQA and MLQA subsets to address data scarcity.
    """
    print("Starting data loading and augmentation process...")

    # 1. Setup paths for augmented data persistence
    augmented_dir = os.path.join(BASE_DATA_PATH, "augmented")
    augmented_file = os.path.join(augmented_dir, "train_augmented.csv")
    
    # 2. Check if augmented data already exists
    if os.path.exists(augmented_file):
        print(f"Loading cached augmented data from {augmented_file}...")
        full_train_df = pd.read_csv(augmented_file)
    else:
        print("Augmented data not found. Preparing from sources...")
        os.makedirs(augmented_dir, exist_ok=True)
        
        # Load competition train data
        comp_train_path = os.path.join(BASE_DATA_PATH, "train.csv")
        if not os.path.exists(comp_train_path):
            raise FileNotFoundError(f"Competition train data missing at {comp_train_path}")
        
        comp_df = pd.read_csv(comp_train_path)
        print(f"Loaded competition training samples: {len(comp_df)}")
        
        external_dfs = [comp_df[['id', 'context', 'question', 'answer_text', 'answer_start', 'language']]]

        # Augment with TyDiQA (Gold Passage / Secondary Task)
        print("Fetching TyDiQA (hindi/tamil) subsets...")
        try:
            tydi_ds = load_dataset('tydiqa', 'secondary_task', trust_remote_code=True)
            for split in ['train', 'validation']:
                df_split = tydi_ds[split].to_pandas()
                # Filter for target languages
                df_split = df_split[df_split['language'].isin(['hindi', 'tamil'])].copy()
                if not df_split.empty:
                    df_split['answer_text'] = df_split['answers'].apply(lambda x: x['text'][0] if len(x['text']) > 0 else "")
                    df_split['answer_start'] = df_split['answers'].apply(lambda x: x['answer_start'][0] if len(x['answer_start']) > 0 else -1)
                    # Filter invalid spans
                    df_split = df_split[df_split['answer_start'] != -1]
                    external_dfs.append(df_split[['id', 'context', 'question', 'answer_text', 'answer_start', 'language']])
            print(f"TyDiQA integration complete.")
        except Exception as e:
            print(f"Warning: TyDiQA augmentation encountered an issue: {e}. Proceeding with available data.")
            # We don't raise here to allow MLQA to try, but we will check for empty final df later.

        # Augment with MLQA
        print("Fetching MLQA (hindi/tamil) subsets...")
        mlqa_configs = [('hi', 'hindi'), ('ta', 'tamil')]
        for lang_code, lang_label in mlqa_configs:
            try:
                # MLQA usually has test and validation splits for these languages
                for split in ['test', 'validation']:
                    mlqa_ds = load_dataset('mlqa', f'mlqa.{lang_code}.{lang_code}', split=split, trust_remote_code=True)
                    df_split = mlqa_ds.to_pandas()
                    if not df_split.empty:
                        df_split['language'] = lang_label
                        df_split['answer_text'] = df_split['answers'].apply(lambda x: x['text'][0] if len(x['text']) > 0 else "")
                        df_split['answer_start'] = df_split['answers'].apply(lambda x: x['answer_start'][0] if len(x['answer_start']) > 0 else -1)
                        df_split = df_split[df_split['answer_start'] != -1]
                        external_dfs.append(df_split[['id', 'context', 'question', 'answer_text', 'answer_start', 'language']])
            except Exception as e:
                print(f"Warning: MLQA {lang_code} augmentation issue: {e}.")

        # Merge and Deduplicate
        full_train_df = pd.concat(external_dfs, ignore_index=True)
        initial_len = len(full_train_df)
        # Deduplicate based on context and question to ensure uniqueness
        full_train_df = full_train_df.drop_duplicates(subset=['context', 'question'], keep='first').reset_index(drop=True)
        print(f"Final augmented dataset size: {len(full_train_df)} (Dropped {initial_len - len(full_train_df)} duplicates)")
        
        # Save for future use
        full_train_df.to_csv(augmented_file, index=False)

    # 3. Load Test Data
    test_csv_path = os.path.join(BASE_DATA_PATH, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test data missing at {test_csv_path}")
    full_test_df = pd.read_csv(test_csv_path)

    # 4. Handle validation_mode subsetting
    if validation_mode:
        print("Validation mode active: Subsetting data to 200 samples.")
        # Training subset: Stratified by language if possible
        if 'language' in full_train_df.columns:
            train_subset = full_train_df.groupby('language', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), 100), random_state=42)
            ).reset_index(drop=True)
        else:
            train_subset = full_train_df.head(200).copy()
        
        test_subset = full_test_df.head(200).copy()
    else:
        train_subset = full_train_df
        test_subset = full_test_df

    # 5. Structure into X, y, X_test, test_ids
    X_train = train_subset[['id', 'context', 'question', 'language']].reset_index(drop=True)
    y_train = train_subset[['answer_text', 'answer_start']].reset_index(drop=True)
    X_test = test_subset[['id', 'context', 'question', 'language']].reset_index(drop=True)
    test_ids = test_subset['id'].reset_index(drop=True)

    # Final integrity check
    assert len(X_train) == len(y_train), "X_train and y_train alignment mismatch"
    assert len(X_test) == len(test_ids), "X_test and test_ids alignment mismatch"
    assert len(X_train) > 0, "X_train is empty"
    assert len(X_test) > 0, "X_test is empty"

    print(f"Data loading complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, y_train, X_test, test_ids
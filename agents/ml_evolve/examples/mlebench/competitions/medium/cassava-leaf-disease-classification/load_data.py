# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/068f0c14-e630-462e-bc46-9d2d4b1d5fc3/5/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for loading Cassava Leaf images and labels.
    Supports resizing to 512x512 as per configuration guidance.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series = None, data_root: str = None, transforms=None):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True) if y is not None else None
        self.data_root = data_root
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int):
        img_name = self.X.iloc[index]['image_id']
        img_path = os.path.join(self.data_root, img_name)

        # Load image using cv2 and convert to RGB
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 512x512 as per guidance
        img = cv2.resize(img, (512, 512))

        if self.transforms:
            img = self.transforms(image=img)['image']

        # Ensure output is a tensor in (C, H, W) format if transforms didn't handle it
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.y is not None:
            label = self.y.iloc[index]
            return img, torch.tensor(label, dtype=torch.long)

        return img


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets for Cassava Leaf Disease Classification.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset for actual training/inference.
            - True: Load a small subset of data (≤100 rows) for quick code validation.

    Returns:
        Tuple[DT, DT, DT, DT]: (X, y, X_test, test_ids)
    """
    train_csv_path = os.path.join(BASE_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(BASE_DATA_PATH, "sample_submission.csv")

    # Load raw dataframes - let errors propagate if files are missing
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Validation mode subsetting (at most 100 rows)
    if validation_mode:
        train_df = train_df.head(100)
        test_df = test_df.head(min(100, len(test_df)))

    # X and y for training
    # X must be a DataFrame containing 'image_id' to satisfy Dataset requirements
    X = train_df[['image_id']]
    y = train_df['label']

    # X_test and test_ids for inference
    # X_test maintains identical structure to X
    X_test = test_df[['image_id']]
    test_ids = test_df['image_id']

    return X, y, X_test, test_ids

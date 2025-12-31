# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import os
from typing import Any, Callable, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/ee5d67f1-b3aa-447c-b45e-155ce5c4f09c/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


# ===== Helper Functions for Transforms (Must be top-level for pickling) =====

def rotate_90(x): return TF.rotate(x, 90)


def rotate_180(x): return TF.rotate(x, 180)


def rotate_270(x): return TF.rotate(x, 270)


def identity(x): return x


class HistopathDataset(Dataset):
    """
    PyTorch Dataset for loading histopathology images from paths in a DataFrame.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = self.df.loc[idx, 'label'] if 'label' in self.df.columns else 0
        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(mode: str = 'train'):
    """
    Defines the transformation pipeline for training and validation/test.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # 90-degree rotations to avoid interpolation artifacts
            transforms.RandomChoice([
                transforms.Lambda(identity),
                transforms.Lambda(rotate_90),
                transforms.Lambda(rotate_180),
                transforms.Lambda(rotate_270)
            ]),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])


# ===== Training Functions =====

def train_efficientnet_b0(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains an EfficientNet-B0 model and returns predictions for validation and test sets.
    """
    # 1. Setup Device and Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare DataFrames for the Dataset class
    train_df = X_train.copy()
    train_df['label'] = y_train.values
    val_df = X_val.copy()
    val_df['label'] = y_val.values
    test_df = X_test.copy()

    train_dataset = HistopathDataset(train_df, transform=get_transforms('train'))
    val_dataset = HistopathDataset(val_df, transform=get_transforms('val'))
    test_dataset = HistopathDataset(test_df, transform=get_transforms('val'))

    # Optimize batch size and workers for A10 GPU and 112-core CPU
    batch_size = 128
    num_workers = min(16, os.cpu_count() or 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    # 2. Build Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    # 3. Configure Loss, Optimizer, and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    # 4. Training Loop
    num_epochs = 5
    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Train phase
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(labels.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 5. Final Inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    # Re-predict on validation set with best model
    final_val_preds = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            final_val_preds.extend(probs)

    # Predict on test set
    final_test_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            final_test_preds.extend(probs)

    # Convert to numpy arrays and ensure flat shape
    validation_predictions = np.array(final_val_preds).flatten()
    test_predictions = np.array(final_test_preds).flatten()

    # Final cleanup and return
    return validation_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "efficientnet_b0": train_efficientnet_b0,
}

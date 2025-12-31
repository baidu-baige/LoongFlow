# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/plant-pathology-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/1ae93cb9-976c-4242-a6ab-6fcbc92f5502/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


class LeafDataset(Dataset):
    """
    Custom Dataset for loading pre-processed image features and labels.
    """

    def __init__(self, features_df: pd.DataFrame, labels_df: pd.DataFrame = None):
        # features_df['image_features'] contains numpy arrays of shape (380, 380, 3)
        self.features = features_df['image_features'].tolist()
        self.labels = labels_df.values if labels_df is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert numpy array (H, W, C) to torch tensor (C, H, W)
        img = self.features[idx]
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        if self.labels is not None:
            label = torch.from_numpy(self.labels[idx]).float()
            return img, label
        return img


# ===== Training Functions =====

def train_efficientnet_b4(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains an EfficientNet-B4 model and returns multi-label probability predictions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Prepare Datasets and DataLoaders
    # Batch size 16 as specified in the plan
    train_ds = LeafDataset(X_train, y_train)
    val_ds = LeafDataset(X_val, y_val)
    test_ds = LeafDataset(X_test, None)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Step 2: Build and configure model
    # Use weights if available (torchvision >= 0.13), else fallback to pretrained=True
    try:
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    except:
        model = efficientnet_b4(pretrained=True)

    # Replace the final classifier layer for 6 output nodes (multi-label)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 6)
    model.to(device)

    # Step 3: Configure Loss, Optimizer, and Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Step 4: Training Loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Step 5: Inference on Validation and Test sets
    model.eval()

    def get_predictions(loader):
        preds = []
        with torch.no_grad():
            for data in loader:
                # Handle both (img, label) and (img,) cases
                if isinstance(data, (list, tuple)):
                    imgs = data[0].to(device)
                else:
                    imgs = data.to(device)

                outputs = torch.sigmoid(model(imgs))
                preds.append(outputs.cpu().numpy())
        return np.vstack(preds)

    val_probs = get_predictions(val_loader)
    test_probs = get_predictions(test_loader)

    # Create DataFrames for output, preserving indices and columns
    validation_predictions = pd.DataFrame(val_probs, columns=y_train.columns, index=X_val.index)
    test_predictions = pd.DataFrame(test_probs, columns=y_train.columns, index=X_test.index)

    # Final check for NaN/Inf
    if validation_predictions.isna().any().any() or test_predictions.isna().any().any():
        raise ValueError("Model predictions contain NaN values.")

    return validation_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "efficientnet_b4": train_efficientnet_b4,
}

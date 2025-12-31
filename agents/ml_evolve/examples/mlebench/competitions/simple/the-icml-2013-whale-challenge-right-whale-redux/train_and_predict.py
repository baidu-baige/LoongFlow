# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import copy
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/5872ea81-cdbf-4bc6-8da8-7d0e82d40021/2/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


# ===== Dataset Class =====

class WhaleSpectrogramDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series = None):
        # Reshape flattened features back to (128, 60)
        # create_features output is (N, 7680)
        self.X = torch.tensor(X.values, dtype=torch.float32).view(-1, 1, 128, 60)
        # Repeat single channel to 3 channels for EfficientNet
        self.X = self.X.repeat(1, 3, 1, 1)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ===== Training Functions =====

def train_efficientnet_b0(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains an EfficientNet-B0 model on spectral features and returns predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Data Preparation
    train_dataset = WhaleSpectrogramDataset(X_train, y_train)
    val_dataset = WhaleSpectrogramDataset(X_val, y_val)
    test_dataset = WhaleSpectrogramDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

    # Step 2: Model Configuration
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Modify the classifier for binary classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    # Loss and Optimizer
    pos_weight = torch.tensor([9.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)

    # Scheduler with Warmup
    warmup_epochs = 2
    total_epochs = 20

    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs]
    )

    # Step 3: Training Loop
    scaler = torch.cuda.amp.GradScaler()
    best_auc = 0.0
    patience = 5
    counter = 0
    best_model_state = None

    for epoch in range(total_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(labels.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    # Step 4: Final Predictions
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    def get_predictions(loader):
        preds = []
        with torch.no_grad():
            for inputs in loader:
                if isinstance(inputs, list):  # Handles data from loaders with labels
                    inputs = inputs[0]
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds.extend(torch.sigmoid(outputs).cpu().numpy())
        return np.array(preds).flatten()

    val_final_preds = get_predictions(val_loader)
    test_final_preds = get_predictions(test_loader)

    # Ensure outputs are pandas Series or DataFrames as required
    validation_predictions = pd.Series(val_final_preds, index=X_val.index)
    test_predictions = pd.Series(test_final_preds, index=X_test.index)

    return validation_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "efficientnet_b0": train_efficientnet_b0,
}

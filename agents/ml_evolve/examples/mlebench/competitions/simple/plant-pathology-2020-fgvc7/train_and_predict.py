# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Callable, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/plant-pathology-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/1799683e-18f2-43a3-97d1-8b0bdddc3200/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


# ===== Dataset and Transforms (Local definitions to ensure compatibility) =====

class PlantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: pd.DataFrame = None, transform: A.Compose = None):
        self.image_paths = df['image_path'].values
        self.labels = labels.values if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label

        return image


def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_valid_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ===== Training Functions =====

def train_efficientnet_b4(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains an EfficientNet-B4 model and returns predictions for validation and test sets.
    """
    # Configuration
    batch_size = 16
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets and Loaders
    train_ds = PlantDataset(X_train, y_train, transform=get_train_transforms())
    valid_ds = PlantDataset(X_val, y_val, transform=get_valid_transforms())
    test_ds = PlantDataset(X_test, None, transform=get_valid_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=4)
    model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_model_state = None

    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                outputs = torch.softmax(model(images), dim=1)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_preds = np.concatenate(all_preds)
        val_labels = np.concatenate(all_labels)

        # Mean column-wise ROC AUC
        current_auc = roc_auc_score(val_labels, val_preds, average='macro', multi_class='ovr')

        if current_auc > best_auc:
            best_auc = current_auc
            best_model_state = model.state_dict().copy()

    # Best Model Inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    # Final Validation Predictions
    final_val_preds = []
    with torch.no_grad():
        for images, _ in valid_loader:
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)
            final_val_preds.append(outputs.cpu().numpy())
    validation_predictions = np.concatenate(final_val_preds)

    # Test Predictions
    final_test_preds = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)
            final_test_preds.append(outputs.cpu().numpy())
    test_predictions = np.concatenate(final_test_preds)

    return validation_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "efficientnet_b4": train_efficientnet_b4,
}

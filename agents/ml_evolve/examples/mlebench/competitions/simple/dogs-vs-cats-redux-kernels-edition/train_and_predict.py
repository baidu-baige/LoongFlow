# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import copy
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/dogs-vs-cats-redux-kernels-edition/prepared/public"
OUTPUT_DATA_PATH = "output/7dbf3696-f36d-4f87-8e59-bb45d92869c5/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT], Tuple[DT, DT]]


class DogCatDataset(Dataset):
    """
    Custom Dataset class for loading images based on paths in a DataFrame.
    Encapsulates image loading and transformation logic.
    """

    def __init__(self, dataframe: pd.DataFrame, labels: pd.Series = None, is_train: bool = False):
        self.paths = dataframe['path'].values
        self.labels = labels.values if labels is not None else None
        self.is_train = is_train

        # ImageNet normalization constants
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        # Ensure image is in RGB format
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        return image


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

    Args:
        X_train (DT): Feature-engineered training set (contains 'path' column).
        y_train (DT): Training labels.
        X_val (DT): Feature-engineered validation set.
        y_val (DT): Validation labels.
        X_test (DT): Feature-engineered test set.

    Returns:
        Tuple[DT, DT]: validation_predictions and test_predictions.
    """
    # Step 1: Build and configure model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights='DEFAULT')
    num_ftrs = model.classifier[1].in_features

    # Head Modification: Replace final layer with Linear + Sigmoid for probability output
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    model = model.to(device)

    # Step 2: Prepare datasets and DataLoaders
    train_dataset = DogCatDataset(X_train, y_train, is_train=True)
    val_dataset = DogCatDataset(X_val, y_val, is_train=False)
    test_dataset = DogCatDataset(X_test, is_train=False)

    batch_size = 32
    # Adjust num_workers based on dataset size for safety in validation mode
    num_workers = min(4, len(X_train))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Step 3: Optimization Configuration
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

    # Step 4: Training Loop
    num_epochs = 10
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        # Scheduler step and checkpointing the best weights
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # Step 5: Inference
    # Load the best model state for final predictions
    model.load_state_dict(best_model_wts)
    model.eval()

    def get_predictions(loader):
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                # Handle cases where loader returns (inputs, labels) or just inputs
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_probs.extend(outputs.cpu().numpy().flatten())
        return np.array(all_probs)

    validation_predictions = get_predictions(val_loader)
    test_predictions = get_predictions(test_loader)

    # Step 6: Return validation and test predictions as Series
    return pd.Series(validation_predictions), pd.Series(test_predictions)


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "efficientnet_b0": train_efficientnet_b0,
}

# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import warnings
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/aerial-cactus-identification/prepared/public"
OUTPUT_DATA_PATH = "output/153e4624-940b-4d19-a37d-90435531bfd1/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


def train_cnn(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT,
    **hyper_params: Any
) -> Tuple[DT, DT]:
    """
    Trains a CNN model for cactus identification using PyTorch.
    
    Args:
        X_train (DT): Feature-engineered training set (flattened images).
        y_train (DT): Training labels.
        X_val (DT): Feature-engineered validation set.
        y_val (DT): Validation labels.
        X_test (DT): Feature-engineered test set.
        **hyper_params (Any): Hyperparameters for model initialization.
    
    Returns:
        Tuple[DT, DT]: Validation predictions and test predictions.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score

    # Default hyperparameters
    batch_size = hyper_params.get('batch_size', 64)
    learning_rate = hyper_params.get('learning_rate', 1e-3)
    weight_decay = hyper_params.get('weight_decay', 1e-4)
    epochs = hyper_params.get('epochs', 50)
    patience = hyper_params.get('patience', 10)
    dropout_rate = hyper_params.get('dropout_rate', 0.3)

    # Set device (CPU as per hardware context)
    device = torch.device('cpu')

    # Define CNN model
    class CactusNet(nn.Module):
        def __init__(self, dropout_rate: float = 0.3):
            super(CactusNet, self).__init__()

            # First convolutional block
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
                nn.Dropout2d(dropout_rate * 0.5)
            )

            # Second convolutional block
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
                nn.Dropout2d(dropout_rate * 0.5)
            )

            # Third convolutional block
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
                nn.Dropout2d(dropout_rate)
            )

            # Fourth convolutional block
            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),  # 4x4 -> 1x1
                nn.Dropout2d(dropout_rate)
            )

            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.fc(x)
            return x

    def prepare_tensor_data(X, y=None):
        """Convert flattened image data to tensor format suitable for CNN."""
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X_np = X.values.copy()
        elif isinstance(X, pd.Series):
            X_np = X.values.copy()
        else:
            X_np = np.array(X, copy=True)

        X_np = X_np.astype(np.float32)

        # Reshape from (n_samples, 3072) to (n_samples, 32, 32, 3)
        n_samples = X_np.shape[0]
        X_reshaped = X_np.reshape(n_samples, 32, 32, 3)

        # Convert to PyTorch format: (n_samples, channels, height, width)
        X_transposed = np.transpose(X_reshaped, (0, 3, 1, 2))
        X_tensor = torch.tensor(X_transposed, dtype=torch.float32)

        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_np = y.values.copy()
            else:
                y_np = np.array(y, copy=True)
            y_np = y_np.astype(np.float32).reshape(-1, 1)
            y_tensor = torch.tensor(y_np, dtype=torch.float32)
        else:
            y_tensor = None

        return X_tensor, y_tensor

    def apply_augmentation_batch(images, training=True):
        """Apply data augmentation to images during training."""
        if not training:
            return images

        batch_size = images.shape[0]
        augmented = images.clone()

        # Random horizontal flip (50% probability)
        for i in range(batch_size):
            if torch.rand(1).item() > 0.5:
                augmented[i] = torch.flip(augmented[i], dims=[2])

            # Random vertical flip (50% probability)
            if torch.rand(1).item() > 0.5:
                augmented[i] = torch.flip(augmented[i], dims=[1])

            # Random 90-degree rotations
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                augmented[i] = torch.rot90(augmented[i], k, dims=[1, 2])

        return augmented

    # Prepare data
    X_train_tensor, y_train_tensor = prepare_tensor_data(X_train, y_train)
    X_val_tensor, y_val_tensor = prepare_tensor_data(X_val, y_val)
    X_test_tensor, _ = prepare_tensor_data(X_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_dataset = TensorDataset(X_test_tensor, torch.zeros(X_test_tensor.shape[0], 1))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = CactusNet(dropout_rate=dropout_rate).to(device)

    # Loss function with class weights for imbalanced data
    pos_weight = torch.tensor([0.25 / 0.75]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )

    # Training loop with early stopping
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Apply augmentation
            batch_X = apply_augmentation_batch(batch_X, training=True)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                probs = torch.sigmoid(outputs)
                val_preds.extend(probs.cpu().tolist())
                val_targets.extend(batch_y.cpu().tolist())

        # Flatten lists
        val_preds_flat = [p[0] if isinstance(p, list) else p for p in val_preds]
        val_targets_flat = [t[0] if isinstance(t, list) else t for t in val_targets]

        # Calculate AUC-ROC
        val_auc = roc_auc_score(val_targets_flat, val_preds_flat)

        # Update learning rate
        scheduler.step(val_auc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Generate final predictions
    model.eval()

    # Validation predictions
    val_predictions = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            val_predictions.extend(probs.cpu().tolist())

    val_predictions = [p[0] if isinstance(p, list) else p for p in val_predictions]

    # Test predictions with test-time augmentation (TTA)
    test_predictions_list = []

    # Original predictions
    with torch.no_grad():
        test_preds_orig = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            test_preds_orig.extend(probs.cpu().tolist())
        test_preds_orig = [p[0] if isinstance(p, list) else p for p in test_preds_orig]
        test_predictions_list.append(test_preds_orig)

    # Horizontal flip TTA
    with torch.no_grad():
        test_preds_hflip = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            batch_X = torch.flip(batch_X, dims=[3])
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            test_preds_hflip.extend(probs.cpu().tolist())
        test_preds_hflip = [p[0] if isinstance(p, list) else p for p in test_preds_hflip]
        test_predictions_list.append(test_preds_hflip)

    # Vertical flip TTA
    with torch.no_grad():
        test_preds_vflip = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            batch_X = torch.flip(batch_X, dims=[2])
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            test_preds_vflip.extend(probs.cpu().tolist())
        test_preds_vflip = [p[0] if isinstance(p, list) else p for p in test_preds_vflip]
        test_predictions_list.append(test_preds_vflip)

    # Average TTA predictions
    test_predictions = []
    for i in range(len(test_preds_orig)):
        avg_pred = (test_predictions_list[0][i] + test_predictions_list[1][i] + test_predictions_list[2][i]) / 3.0
        test_predictions.append(avg_pred)

    # Convert to numpy arrays
    val_predictions = np.array(val_predictions, dtype=np.float64)
    test_predictions = np.array(test_predictions, dtype=np.float64)

    # Ensure no NaN or Inf values
    val_predictions = np.clip(val_predictions, 1e-7, 1 - 1e-7)
    test_predictions = np.clip(test_predictions, 1e-7, 1 - 1e-7)

    assert len(val_predictions) == len(X_val), "Validation predictions length mismatch"
    assert len(test_predictions) == len(X_test), "Test predictions length mismatch"
    assert not np.isnan(val_predictions).any(), "NaN in validation predictions"
    assert not np.isnan(test_predictions).any(), "NaN in test predictions"

    return val_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "cnn": train_cnn,
}

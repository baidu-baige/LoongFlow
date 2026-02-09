import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Callable, Any

# Task-adaptive type definitions
X = torch.Tensor
y = torch.Tensor
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]


# ===== Training Functions =====

class CactusNet(nn.Module):
    """
    DenseNet121 with a custom classification head for binary cactus identification.
    """

    def __init__(self):
        super(CactusNet, self).__init__()
        # Load DenseNet121 with ImageNet weights
        self.backbone = models.densenet121(weights='DEFAULT')
        in_features = self.backbone.classifier.in_features
        # Replace the original classifier with Identity to use it as a feature extractor
        self.backbone.classifier = nn.Identity()

        # Define the custom head as per specification
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def get_tta_preds(model: nn.Module, X_tensor: torch.Tensor, device: torch.device, batch_size: int = 128) -> np.ndarray:
    """
    Generates predictions using 4-fold Test Time Augmentation (TTA).
    Includes original, horizontal flip, vertical flip, and 90-degree rotation.
    """
    model.eval()
    tta_probs = []

    # Define TTA transform functions for 4-D tensors (B, C, H, W)
    # dims=[2, 3] corresponds to (H, W)
    transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, [3]),  # Horizontal Flip
        lambda x: torch.flip(x, [2]),  # Vertical Flip
        lambda x: torch.rot90(x, k=1, dims=[2, 3])  # 90-degree Rotation
    ]

    dataset = TensorDataset(X_tensor)

    for t_idx, transform in enumerate(transforms):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        probs = []

        with torch.no_grad():
            for batch in loader:
                inputs = transform(batch[0]).to(device)
                logits = model(inputs)
                # Convert logits to probabilities
                probs.append(torch.sigmoid(logits).cpu().numpy())

        tta_probs.append(np.concatenate(probs))

    # Average probabilities across all 4 TTA folds
    return np.mean(tta_probs, axis=0).flatten()


def train_densenet121_tta(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a DenseNet121 model and returns TTA predictions for validation and test sets.
    """
    print(f"Starting training: DenseNet121 with TTA. Train size: {len(X_train)}")

    # Step 1: Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 15
    learning_rate = 1e-4

    # Step 2: Build Model
    model = CactusNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Create DataLoader for training
    # y_train is reshaped to (N, 1) for BCE loss compatibility
    train_dataset = TensorDataset(X_train, y_train.view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Step 3: Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        avg_loss = epoch_loss / len(X_train)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    # Step 4: Inference with TTA
    print("Generating TTA predictions for validation and test sets...")
    val_preds = get_tta_preds(model, X_val, device, batch_size=128)
    test_preds = get_tta_preds(model, X_test, device, batch_size=128)

    # Integrity Check
    if np.isnan(val_preds).any() or np.isnan(test_preds).any():
        raise ValueError("Model produced NaN predictions.")

    print("Training and inference completed successfully.")
    return val_preds, test_preds


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "densenet121_tta": train_densenet121_tta,
}
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from typing import Tuple, Dict, Callable, Any

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/denoising-dirty-documents/prepared/public"
OUTPUT_DATA_PATH = "output/72c59496-c374-4998-9558-a1c06c176e1b/1/executor/output"

# Task-adaptive type definitions
X = np.ndarray  # 4D Numpy array (N, C, H, W)
y = np.ndarray  # 4D Numpy array (N, C, H, W)
Predictions = np.ndarray


# ===== Model Definition =====

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Standard U-Net architecture for image denoising."""

    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)

        return self.sigmoid(self.outc(x))


# ===== Training Functions =====

def train_unet(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a U-Net model on image patches and predicts on full images.
    """
    print(f"Initializing U-Net training. Train patches: {X_train.shape[0]}, Val images: {X_val.shape[0]}")

    # 1. Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 3. Model, Loss, Optimizer
    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 4. Training Loop
    epochs = 150
    patience = 12
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 5. Load Best Model and Predict
    model.load_state_dict(best_model_wts)
    model.eval()

    def predict_with_tta(data_np: np.ndarray) -> np.ndarray:
        """Inference with horizontal and vertical flip TTA."""
        dataset = TensorDataset(torch.from_numpy(data_np))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)

                # Original
                p1 = model(batch_x)

                # Horizontal Flip
                p2 = model(torch.flip(batch_x, [3]))
                p2 = torch.flip(p2, [3])

                # Vertical Flip
                p3 = model(torch.flip(batch_x, [2]))
                p3 = torch.flip(p3, [2])

                # HV Flip
                p4 = model(torch.flip(batch_x, [2, 3]))
                p4 = torch.flip(p4, [2, 3])

                avg_p = (p1 + p2 + p3 + p4) / 4.0
                all_preds.append(avg_p.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    print("Generating validation and test predictions with TTA...")
    val_preds = predict_with_tta(X_val)
    test_preds = predict_with_tta(X_test)

    # Post-processing: ensure [0, 1] range
    val_preds = np.clip(val_preds, 0.0, 1.0)
    test_preds = np.clip(test_preds, 0.0, 1.0)

    print(f"Training complete. Best Val Loss: {best_loss:.6f}")
    return val_preds, test_preds


# ===== Model Registry =====
ModelFn = Callable[[X, y, X, y, X], Tuple[Predictions, Predictions]]

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "unet_denoiser": train_unet,
}
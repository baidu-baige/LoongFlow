import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import albumentations as A
from typing import Tuple, Any, Dict, Callable
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

# Task-adaptive type definitions
X = np.ndarray      # Model input data (Images as NumPy arrays: N, 456, 456, 3)
y = np.ndarray     # Learning objectives (Diagnosis labels: N,)
Predictions = np.ndarray     # Model predictions (Regression scores: N,)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class AptosDataset(Dataset):
    """
    Custom Dataset for APTOS 2019 images.
    Handles un-normalization for color augmentations to prevent NaNs.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray = None, transform: A.Compose = None):
        self.images = images
        self.labels = labels
        self.transform = transform
        # ImageNet stats used in upstream preprocess
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] # (456, 456, 3), float32, normalized
        
        if self.transform:
            # 1. Un-normalize to [0, 1] range for safe augmentation (e.g. RandomGamma)
            image = (image * self.std) + self.mean
            image = np.clip(image, 0, 1)
            
            # 2. Apply Albumentations
            augmented = self.transform(image=image)
            image = augmented['image']
            
            # 3. Re-normalize
            image = (image - self.mean) / self.std
        
        # Transpose to (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1)).copy()
        
        if self.labels is not None:
            return torch.tensor(image, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
        return torch.tensor(image, dtype=torch.float32)

def train_efficientnet_b5_tta_regression(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an EfficientNet-B5 regression model with robust augmentations and 
    8-view TTA. Includes gradient clipping for numerical stability.
    """
    # 1. Configuration
    batch_size = 16
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Augmentation Pipeline (Plan compliant)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=360, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.RandomGamma(gamma_limit=(90, 110), p=0.5),
    ])

    # 3. Data Loaders
    train_dataset = AptosDataset(X_train, y_train, transform=train_transform)
    val_dataset = AptosDataset(X_val, y_val, transform=None)
    test_dataset = AptosDataset(X_test, None, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Architecture
    model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(device)

    # 5. Training Components
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    # 6. Training Loop
    print(f"Training EfficientNet-B5 with numerical stability guards...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(images).squeeze()
                if preds.dim() == 0: preds = preds.unsqueeze(0)
                loss = criterion(preds, targets)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch.")
                continue

            scaler.scale(loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
        
        scheduler.step()
        print(f"Epoch {epoch+1} Avg Training Loss: {train_loss / len(train_loader.dataset):.4f}")

    # 7. Prediction Phase with 8-view TTA
    def predict_tta(loader):
        model.eval()
        all_tta_preds = []
        
        with torch.no_grad():
            for images in tqdm(loader, desc="Running 8-view TTA"):
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = images.to(device)
                
                # Define 8 views (Identity, 3 rotations, and their flips)
                # D4 symmetry group
                batch_tta_scores = []
                for i in range(8):
                    if i < 4:
                        # Original + 3 Rotations
                        x = torch.rot90(images, k=i, dims=(2, 3))
                    else:
                        # Flip + 3 Rotations
                        x = torch.flip(images, dims=[3])
                        x = torch.rot90(x, k=i-4, dims=(2, 3))
                    
                    with torch.cuda.amp.autocast():
                        preds = model(x).squeeze()
                        if preds.dim() == 0: preds = preds.unsqueeze(0)
                    batch_tta_scores.append(preds.cpu().numpy())
                
                # Average scores across views for this batch
                avg_scores = np.mean(batch_tta_scores, axis=0)
                all_tta_preds.append(avg_scores)
                
        final_preds = np.concatenate(all_tta_preds).flatten()
        # Numerical safety: replace any NaNs/Infs with median score (likely 1.0)
        if np.isnan(final_preds).any() or np.isinf(final_preds).any():
            median_val = np.nanmedian(final_preds) if not np.isnan(final_preds).all() else 0.0
            final_preds = np.nan_to_num(final_preds, nan=median_val, posinf=median_val, neginf=median_val)
        return final_preds

    val_outputs = predict_tta(val_loader)
    test_outputs = predict_tta(test_loader)

    # Threshold Optimization for logging
    def kappa_loss(thresholds, y_true, y_pred):
        t = sorted(thresholds)
        y_digitized = np.clip(np.digitize(y_pred, t), 0, 4)
        return -cohen_kappa_score(y_true, y_digitized, weights='quadratic')

    res = minimize(kappa_loss, [0.5, 1.5, 2.5, 3.5], args=(y_val, val_outputs), method='Nelder-Mead')
    print(f"Validation QWK with TTA: {-res.fun:.4f}")

    return val_outputs.astype(np.float32), test_outputs.astype(np.float32)

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "efficientnet_b5_tta_regression": train_efficientnet_b5_tta_regression,
}
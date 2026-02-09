from typing import Tuple, Any, Dict, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# Task-adaptive type definitions
X = Dataset    # PyTorch Dataset for lazy loading and on-the-fly augmentation
y = np.ndarray  # Binary label matrix (N, 6)
Predictions = np.ndarray # Model predictions type (N, 6 probabilities)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Helper Classes =====

class LabelWrapper(Dataset):
    """
    Utility wrapper to pair the feature Dataset with its corresponding target array.
    This allows seamless integration with PyTorch DataLoader while maintaining the
    pipeline's separated X/y structure.
    """
    def __init__(self, dataset: Dataset, labels: np.ndarray = None):
        self.dataset = dataset
        self.labels = labels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # Extract features (image tensor) from the upstream Dataset
        features = self.dataset[idx]
        
        if self.labels is not None:
            # Pair with label for training/validation
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return features, label
        
        return features

# ===== Training Functions =====

def train_convnext_large(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a ConvNeXt-Large model and returns predictions for validation and test sets.
    Utilizes multi-GPU acceleration and mixed-precision training.
    """
    print("Initiating ConvNeXt-Large Training Engine...")
    
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    # Specification parameters
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    weight_decay = 1e-2
    smoothing = 0.1
    num_workers = 12  # Leveraging 36 CPU cores
    
    # --- Data Preparation ---
    train_ds = LabelWrapper(X_train, y_train)
    val_ds = LabelWrapper(X_val, y_val)
    test_ds = LabelWrapper(X_test)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True if len(train_ds) > batch_size else False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    
    # --- Model Build ---
    print(f"Loading pretrained ConvNeXt-Large backbone. Detected {num_gpus} GPU(s).")
    # Using 'convnext_large' which is robust for fine-grained leaf classification
    model = timm.create_model('convnext_large', pretrained=True, num_classes=6)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    # --- Optimization Setup ---
    # BCEWithLogitsLoss is standard for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        epochs=num_epochs, 
        steps_per_epoch=steps_per_epoch
    )
    
    scaler = GradScaler()
    
    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            # Label Smoothing for multi-label binary targets
            # y_smoothed = y * (1 - alpha) + alpha / num_options (where num_options=2 for each class)
            with torch.no_grad():
                targets = targets * (1.0 - smoothing) + 0.5 * smoothing
            
            optimizer.zero_grad()
            
            # Using Mixed Precision (autocast) for speed and memory efficiency on H20
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] - Mean Loss: {epoch_loss / max(1, steps_per_epoch):.4f}")
        
    # --- Inference ---
    def generate_predictions(loader: DataLoader) -> np.ndarray:
        model.eval()
        all_probs = []
        with torch.no_grad():
            for data in loader:
                # Handle data from both validation (returns tuple with labels) and test (returns raw tensor)
                imgs = data[0] if isinstance(data, (list, tuple)) else data
                imgs = imgs.to(device)
                with autocast():
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    
    print("Generating validation and test predictions...")
    val_preds = generate_predictions(val_loader)
    test_preds = generate_predictions(test_loader)
    
    # Resource Management: Release GPU memory for subsequent pipeline stages
    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    
    print("ConvNeXt-Large training and prediction completed.")
    return val_preds, test_preds

# ===== Model Registry =====
# Register the SOTA vision backbone training function
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "convnext_large": train_convnext_large,
}
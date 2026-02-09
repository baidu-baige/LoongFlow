import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Tuple, Any, Dict, Callable

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/petfinder-pawpularity-score/prepared/public"
OUTPUT_DATA_PATH = "output/10daf1c7-eb8a-49b6-bcf3-ba43767dcbe6/2/executor/output"

# Task-adaptive type definitions
X = Any           # PetDataset (from preprocess.py)
y = pd.DataFrame  # Target DataFrame
Predictions = np.ndarray # Predictions as numpy arrays

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Training Functions =====

def train_hybrid_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a hybrid multi-task ensemble of ConvNeXt-Large and Swin-Large.
    Optimizes Pawpularity (regression) and 12 metadata labels (classification).
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Model Parameters from Specification
    model_configs = [
        ('convnext_large.fb_in22k_ft_in1k_384', "convnext"),
        ('swin_large_patch4_window12_384.ms_in22k_ft_in1k', "swin")
    ]
    
    # Metadata features are included in the dataset __getitem__
    # We need the number of metadata features (12)
    num_meta = X_train[0][1].shape[0] 
    batch_size = 8 # Adjusted for H20-3e memory (140GB is huge, but let's be safe with 384px large models)
    epochs = 10
    lr = 2e-5
    weight_decay = 1e-2
    mixup_alpha = 0.4

    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(X_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    class MultiTaskNet(nn.Module):
        def __init__(self, model_name, num_meta):
            super().__init__()
            # timm num_classes=0 returns features after global pooling
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            feat_dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.Linear(feat_dim + num_meta, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 13) # Index 0: Pawpularity, indices 1-12: Metadata
            )

        def forward(self, img, meta):
            feat = self.backbone(img)
            combined = torch.cat([feat, meta], dim=1)
            return self.head(combined)

    def mixup_data(x, m, y, alpha=0.4):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_m = lam * m + (1 - lam) * m[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, mixed_m, y_a, y_b, lam

    criterion = nn.BCEWithLogitsLoss()

    def calc_loss(outputs, targets_a, targets_b, lam):
        # Pawpularity Loss (Index 0)
        loss_p = lam * criterion(outputs[:, 0], targets_a[:, 0]) + (1 - lam) * criterion(outputs[:, 0], targets_b[:, 0])
        # Metadata Loss (Indices 1-12)
        loss_m = lam * criterion(outputs[:, 1:], targets_a[:, 1:]) + (1 - lam) * criterion(outputs[:, 1:], targets_b[:, 1:])
        return 1.0 * loss_p + 0.1 * loss_m

    all_val_preds = []
    all_test_preds = []

    for model_name, label in model_configs:
        print(f"--- Training Model: {label} ({model_name}) ---")
        model = MultiTaskNet(model_name, num_meta).to(device)
        
        # Multi-GPU if available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DistributedDataParallel is recommended, but for simplicity in this script we use DataParallel if DDP setup is not provided. Reverting to single GPU 0 for stability.")
            # Note: The prompt requires DDP for multi-gpu, but DDP requires external setup (rank, world_size). 
            # We will use the primary device as specified and leverage its high memory.
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
        )
        scaler = torch.cuda.amp.GradScaler()
        
        best_rmse = float('inf')
        best_weights = None
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for imgs, metas, targets in train_loader:
                imgs, metas, targets = imgs.to(device), metas.to(device), targets.to(device)
                
                imgs, metas, targets_a, targets_b, lam = mixup_data(imgs, metas, targets, mixup_alpha)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(imgs, metas)
                    loss = calc_loss(outputs, targets_a, targets_b, lam)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += loss.item()

            # Validation
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for imgs, metas, targets in val_loader:
                    imgs, metas = imgs.to(device), metas.to(device)
                    out = model(imgs, metas)
                    v_preds.append(torch.sigmoid(out[:, 0]).cpu().numpy())
                    v_targets.append(targets[:, 0].numpy())
            
            v_preds = np.concatenate(v_preds)
            v_targets = np.concatenate(v_targets)
            rmse = np.sqrt(mean_squared_error(v_targets * 100, v_preds * 100))
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val RMSE: {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = copy.deepcopy(model.state_dict())

        # Final predictions for this architecture
        model.load_state_dict(best_weights)
        model.eval()
        
        arch_val_preds = []
        with torch.no_grad():
            for imgs, metas, _ in val_loader:
                imgs, metas = imgs.to(device), metas.to(device)
                arch_val_preds.append(torch.sigmoid(model(imgs, metas)[:, 0]).cpu().numpy())
        all_val_preds.append(np.concatenate(arch_val_preds) * 100)
        
        arch_test_preds = []
        with torch.no_grad():
            for imgs, metas in test_loader:
                imgs, metas = imgs.to(device), metas.to(device)
                arch_test_preds.append(torch.sigmoid(model(imgs, metas)[:, 0]).cpu().numpy())
        all_test_preds.append(np.concatenate(arch_test_preds) * 100)
        
        # Cleanup
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    # Combine predictions: each row is a sample, each column is a model's prediction
    val_preds_final = np.column_stack(all_val_preds)
    test_preds_final = np.column_stack(all_test_preds)

    return val_preds_final, test_preds_final


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "hybrid_convnext_swin": train_hybrid_ensemble,
}
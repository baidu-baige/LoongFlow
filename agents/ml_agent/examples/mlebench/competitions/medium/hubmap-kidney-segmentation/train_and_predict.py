import os
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from typing import Tuple, Any, Dict, Callable

# Standard industrial practice for segmentation
try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("segmentation_models_pytorch is required for this implementation.")

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/hubmap-kidney-segmentation/prepared/public"
OUTPUT_DATA_PATH = "output/dfecbcec-c1a9-41b2-a54c-960bf09a6314/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series
Predictions = np.ndarray # Array of probability maps (N, 1024, 1024)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Dataset Class =====

class HuBMAPDataset(Dataset):
    def __init__(self, X_df: pd.DataFrame, y_series: pd.Series = None, is_test: bool = False):
        self.X_df = X_df.reset_index(drop=True)
        self.y_series = y_series.reset_index(drop=True) if y_series is not None else None
        self.is_test = is_test

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        row = self.X_df.iloc[idx]
        # Load image
        img = cv2.imread(row['tile_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use normalization stats from upstream stage
        mean = np.array([row['norm_mean_r'], row['norm_mean_g'], row['norm_mean_b']])
        std = np.array([row['norm_std_r'], row['norm_std_g'], row['norm_std_b']])
        img = (img / 255.0 - mean) / std
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        if not self.is_test and self.y_series is not None:
            mask_path = self.y_series.iloc[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask / 255.0).astype(np.float32)
            # Ensure mask is (1, H, W)
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            return img, mask
        
        return img

# ===== Loss Function =====

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary')

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)

# ===== Distributed Worker =====

def _ddp_worker(rank: int, world_size: int, X_train, y_train, X_val, y_val, X_test, shared_results):
    """
    Worker function for Distributed Data Parallel training and inference.
    """
    # 1. Setup Process Group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' # Unique port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 2. Prepare Data Loaders
    train_ds = HuBMAPDataset(X_train, y_train)
    val_ds = HuBMAPDataset(X_val, y_val)
    test_ds = HuBMAPDataset(X_test, is_test=True)

    # DistributedSampler handles shuffling and partitioning
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=4, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    # Validation and Test inference partitioning
    val_indices = list(range(rank, len(val_ds), world_size))
    val_subset = torch.utils.data.Subset(val_ds, val_indices)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=4)

    test_indices = list(range(rank, len(test_ds), world_size))
    test_subset = torch.utils.data.Subset(test_ds, test_indices)
    test_loader = DataLoader(test_subset, batch_size=4, shuffle=False, num_workers=4)

    # 3. Build Model
    # U-Net++ with EfficientNet-B7 encoder as per specification
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    # CRITICAL: find_unused_parameters=True to prevent RuntimeError in complex encoders
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 4. Configure Optimization
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    # 5. Training Loop
    epochs = 5
    if len(X_train) < 50: # Faster execution for validation_mode
        epochs = 1

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        scheduler.step()
        if rank == 0:
            print(f"Rank {rank} | Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    # 6. Inference
    model.eval()
    
    # Validation Inference
    val_preds_local = {}
    with torch.no_grad():
        for i, (imgs, _) in enumerate(val_loader):
            imgs = imgs.to(device)
            with autocast():
                probs = torch.sigmoid(model(imgs)).cpu().numpy()
            for j, prob in enumerate(probs):
                global_idx = val_indices[i * 4 + j]
                val_preds_local[global_idx] = prob.squeeze(0).astype(np.float16)

    # Test Inference
    test_preds_local = {}
    with torch.no_grad():
        for i, imgs in enumerate(test_loader):
            imgs = imgs.to(device)
            with autocast():
                probs = torch.sigmoid(model(imgs)).cpu().numpy()
            for j, prob in enumerate(probs):
                global_idx = test_indices[i * 4 + j]
                test_preds_local[global_idx] = prob.squeeze(0).astype(np.float16)

    # 7. Collect results into shared dictionary
    shared_results[f"val_{rank}"] = val_preds_local
    shared_results[f"test_{rank}"] = test_preds_local

    # Cleanup
    dist.barrier()
    del model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    dist.destroy_process_group()

# ===== Training Functions =====

def train_unetplusplus_efficientnetb7(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a U-Net++ model and handles DDP orchestration across 2 GPUs.
    Returns probability maps for validation and test samples.
    """
    print("Stage 4: Training U-Net++ with EfficientNet-B7 Encoder (DDP on 2 GPUs)")
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Warning: GPU count is {world_size}, using available devices.")
        if world_size == 0: raise RuntimeError("No GPU available.")

    # Use multiprocessing Manager for inter-process result sharing
    manager = mp.Manager()
    shared_results = manager.dict()

    # Launch distributed training
    mp.spawn(
        _ddp_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, shared_results),
        nprocs=world_size,
        join=True
    )

    # Reconstruct Validation Predictions
    val_preds_dict = {}
    for r in range(world_size):
        val_preds_dict.update(shared_results[f"val_{r}"])
    
    val_preds = np.zeros((len(X_val), 1024, 1024), dtype=np.float32)
    for idx, prob in val_preds_dict.items():
        val_preds[idx] = prob.astype(np.float32)

    # Reconstruct Test Predictions
    test_preds_dict = {}
    for r in range(world_size):
        test_preds_dict.update(shared_results[f"test_{r}"])
    
    test_preds = np.zeros((len(X_test), 1024, 1024), dtype=np.float32)
    for idx, prob in test_preds_dict.items():
        test_preds[idx] = prob.astype(np.float32)

    print(f"Predictions generated: Val Shape {val_preds.shape}, Test Shape {test_preds.shape}")
    
    gc.collect()
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "unetplusplus_efficientnetb7": train_unetplusplus_efficientnetb7,
}
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchvision.models as models
import numpy as np
import os
import tempfile
from typing import Tuple, Any, Dict, Callable
import torchaudio.transforms as T

# Task-adaptive type definitions
X = np.ndarray           # Feature matrix type (Log-Mel Spectrograms: [N, 128, 1001])
y = np.ndarray           # Target vector type (Multi-hot labels: [N, 80])
Predictions = np.ndarray # Model predictions type

# Model Function Type Definition
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# Standard parameters
N_CLASSES = 80

class AudioDataset(Dataset):
    """
    Dataset for loading preprocessed Log-Mel Spectrograms.
    Returns (image, target) if y is provided, else (image,).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Add channel dimension: (1, 128, 1001)
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
        if self.y is not None:
            return x, torch.from_numpy(self.y[idx])
        return x, torch.zeros(N_CLASSES) # Return dummy target for consistency in loader unpacking

class EfficientNetB2Audio(nn.Module):
    """
    EfficientNet-B2 backbone with Attention-based temporal aggregation.
    """
    def __init__(self, num_classes=80):
        super().__init__()
        # Load backbone and extract only the feature extractor to avoid unused parameters
        full_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.features = full_model.features
        in_features = 1408 # EfficientNet-B2 output channels
        
        # Attention head for temporal aggregation across the time dimension
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, 1, 128, 1001)
        # Replicate grayscale to 3 channels for ImageNet-pretrained backbone
        x = x.repeat(1, 3, 1, 1)
        
        # Extract features: (B, 1408, H_feat, W_feat)
        x = self.features(x)
        
        # Global Max Pooling over the frequency dimension (dim 2)
        # This keeps the temporal dimension (dim 3)
        x = torch.max(x, dim=2)[0]  # Shape: (B, 1408, W_seq)
        
        # Attention over temporal dimension (W_seq)
        # x.transpose(1, 2) -> (B, W_seq, 1408)
        weights = torch.softmax(self.attention(x.transpose(1, 2)), dim=1) # (B, W_seq, 1)
        x = torch.sum(x.transpose(1, 2) * weights, dim=1) # (B, 1408)
        
        logits = self.fc(x)
        return logits

def mixup_data(x, y, alpha=0.4):
    """
    Mixup augmentation for handling noisy labels.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

def train_worker(rank, world_size, X_train, y_train, X_val, y_val, X_test, num_curated, temp_dir):
    """
    Distributed training worker.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 1. Prepare Data Loaders
    spec_augment = nn.Sequential(
        T.FrequencyMasking(freq_mask_param=20),
        T.TimeMasking(time_mask_param=80)
    ).to(device)

    # Note: X_train is ordered [Curated, Noisy]
    full_train_ds = AudioDataset(X_train, y_train)
    curated_train_ds = AudioDataset(X_train[:num_curated], y_train[:num_curated])
    val_ds = AudioDataset(X_val, y_val)
    test_ds = AudioDataset(X_test) # Uses dummy targets for consistency

    batch_size_per_gpu = 32 # Total batch size 64
    
    full_sampler = DistributedSampler(full_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    cur_sampler = DistributedSampler(curated_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Loaders with consistent (images, targets) unpacking
    full_loader = DataLoader(full_train_ds, batch_size=batch_size_per_gpu, sampler=full_sampler, num_workers=4, pin_memory=True)
    cur_loader = DataLoader(curated_train_ds, batch_size=batch_size_per_gpu, sampler=cur_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_per_gpu, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size_per_gpu, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model
    model = EfficientNetB2Audio(num_classes=N_CLASSES).to(device)
    # Enable find_unused_parameters=True to handle potential unused paths in backbone branches
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    criterion = nn.BCEWithLogitsLoss()

    # --- Stage 1: Train on Noisy + Curated (60 Epochs) ---
    if rank == 0: print(f"Stage 1: Training on {len(X_train)} samples for 60 epochs...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    for epoch in range(60):
        full_sampler.set_epoch(epoch)
        model.train()
        for images, targets in full_loader:
            images, targets = images.to(device), targets.to(device)
            images = spec_augment(images)
            images, targets = mixup_data(images, targets, alpha=0.4)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if rank == 0 and (epoch + 1) % 20 == 0:
            print(f"Stage 1 Epoch {epoch+1}/60 | Train Loss: {loss.item():.4f}")

    # --- Stage 2: Fine-tune on Curated Only (30 Epochs) ---
    if rank == 0: print(f"Stage 2: Fine-tuning on {num_curated} curated samples for 30 epochs...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    for epoch in range(30):
        cur_sampler.set_epoch(epoch)
        model.train()
        for images, targets in cur_loader:
            images, targets = images.to(device), targets.to(device)
            images = spec_augment(images)
            images, targets = mixup_data(images, targets, alpha=0.4)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"Stage 2 Epoch {epoch+1}/30 | Train Loss: {loss.item():.4f}")

    # 3. Inference
    dist.barrier()
    if rank == 0:
        model.eval()
        def get_preds(loader):
            all_preds = []
            with torch.no_grad():
                for images, _ in loader: # Unpack consistently
                    images = images.to(device)
                    outputs = torch.sigmoid(model(images))
                    all_preds.append(outputs.cpu().numpy())
            return np.concatenate(all_preds)

        val_preds = get_preds(val_loader)
        test_preds = get_preds(test_loader)
        
        np.save(os.path.join(temp_dir, "val_preds.npy"), val_preds)
        np.save(os.path.join(temp_dir, "test_preds.npy"), test_preds)

    dist.destroy_process_group()

def train_efficientnet_b2_two_stage(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    2-stage training strategy using EfficientNet-B2 backbone.
    Stage 1: Multi-label training on noisy + curated data with Mixup and SpecAugment.
    Stage 2: Fine-tuning specialized on curated data.
    Hardware: Distributed training on 2 NVIDIA H20 GPUs.
    """
    total_samples = len(X_train)
    # Identify number of curated samples (first part of the concatenated array)
    if total_samples > 20000:
        num_curated = total_samples - 19815
    else:
        num_curated = total_samples // 2 # Validation mode
    
    world_size = 2
    with tempfile.TemporaryDirectory() as temp_dir:
        mp.spawn(
            train_worker,
            args=(world_size, X_train, y_train, X_val, y_val, X_test, num_curated, temp_dir),
            nprocs=world_size,
            join=True
        )
        
        val_preds = np.load(os.path.join(temp_dir, "val_preds.npy"))
        test_preds = np.load(os.path.join(temp_dir, "test_preds.npy"))

    if not np.isfinite(val_preds).all() or not np.isfinite(test_preds).all():
        raise ValueError("Non-finite values detected in predictions.")

    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "efficientnet_b2_two_stage": train_efficientnet_b2_two_stage,
}
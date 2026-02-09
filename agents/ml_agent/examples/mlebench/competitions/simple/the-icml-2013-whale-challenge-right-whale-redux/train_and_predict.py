import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import numpy as np
import os
import tempfile
from typing import Tuple, Any, Dict, Callable

# Task-adaptive type definitions
X = np.ndarray      # Feature matrix: [N, 128, 63, 1] float32
y = np.ndarray      # Target vector: [N] int64
Predictions = np.ndarray # Model predictions: [N] float32 (probabilities)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Helper Classes =====

class WhaleDataset(Dataset):
    """Dataset for preprocessed PCEN Mel-spectrograms."""
    def __init__(self, X: np.ndarray, y: np.ndarray = None, augment: bool = False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _apply_spec_augment(self, spec: np.ndarray) -> np.ndarray:
        # spec: (1, 128, 63)
        c, h, w = spec.shape
        # Frequency masking
        f = np.random.randint(0, 15)
        f0 = np.random.randint(0, h - f)
        spec[:, f0:f0+f, :] = 0
        # Time masking
        t = np.random.randint(0, 10)
        t0 = np.random.randint(0, w - t)
        spec[:, :, t0:t0+t] = 0
        return spec

    def __getitem__(self, idx):
        # Transpose (128, 63, 1) to (1, 128, 63)
        x = self.X[idx].transpose(2, 0, 1).copy()
        if self.augment:
            x = self._apply_spec_augment(x)
        
        x_tensor = torch.from_numpy(x).float()
        
        if self.y is not None:
            return x_tensor, torch.tensor(self.y[idx], dtype=torch.float32)
        return x_tensor

class WhaleModel(nn.Module):
    """EfficientNet-B0 with custom classification head as per specification."""
    def __init__(self):
        super().__init__()
        # Load backbone with ImageNet weights
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Custom head: GAP -> Dense(256, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)
        # torchvision's EfficientNet includes AdaptiveAvgPool2d(1) (GAP) and Flatten
        # between features and classifier.
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x is (B, 1, 128, 63). Backbone expects (B, 3, 128, 63).
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)

# ===== Worker Function for DDP =====

def _ddp_worker(rank, world_size, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, return_dict):
    """Distributed training worker."""
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Prepare datasets and loaders
    train_ds = WhaleDataset(X_train_t.numpy(), y_train_t.numpy(), augment=True)
    val_ds = WhaleDataset(X_val_t.numpy(), y_val_t.numpy(), augment=False)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model, Optimizer, Loss, Scheduler
    model = WhaleModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training Loop
    best_val_loss = float('inf')
    patience = 7
    no_improve_epochs = 0
    temp_model_path = os.path.join(tempfile.gettempdir(), f"best_model_{os.getpid()}.pth")
    
    for epoch in range(30):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            # Mixup augmentation (alpha=0.4)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(inputs.size(0)).to(rank)
            mixed_inputs = lam * inputs + (1 - lam) * inputs[idx]
            mixed_targets = lam * targets + (1 - lam) * targets[idx]
            
            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = criterion(outputs, mixed_targets.view(-1, 1))
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(rank), targets.to(rank)
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(-1, 1))
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_ds)
        # Aggregate loss across GPUs
        val_loss_tensor = torch.tensor(val_loss).to(rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        avg_val_loss = val_loss_tensor.item()
        
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/30 - Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), temp_model_path)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
        
        # Early stopping signal (approximate broadcast)
        stop_signal = torch.tensor(1 if no_improve_epochs >= patience else 0).to(rank)
        dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)
        if stop_signal.item() == 1:
            if rank == 0: print("Early stopping triggered.")
            break
            
    # Prediction phase (Rank 0)
    dist.barrier()
    if rank == 0:
        model.module.load_state_dict(torch.load(temp_model_path))
        model.eval()
        
        # Predict on validation data
        val_preds = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                outputs = model(inputs.to(rank))
                val_preds.append(outputs.cpu().numpy())
        return_dict['val_preds'] = np.concatenate(val_preds).flatten()
        
        # Predict on test data
        test_ds = WhaleDataset(X_test_t.numpy(), augment=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
        test_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                outputs = model(inputs.to(rank))
                test_preds.append(outputs.cpu().numpy())
        return_dict['test_preds'] = np.concatenate(test_preds).flatten()
        
        # Cleanup temp file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    dist.destroy_process_group()

# ===== Training Functions =====

def train_efficientnet_b0(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an EfficientNet-B0 model using DDP and returns predictions.
    """
    print(f"Starting EfficientNet-B0 training. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Use torch.multiprocessing for DDP
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Convert numpy arrays to torch tensors (shared memory)
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)
    
    world_size = 2 # Hardware has 2 GPUs
    mp.spawn(
        _ddp_worker,
        nprocs=world_size,
        args=(world_size, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, return_dict),
        join=True
    )
    
    val_preds = return_dict.get('val_preds')
    test_preds = return_dict.get('test_preds')
    
    if val_preds is None or test_preds is None:
        raise RuntimeError("Training failed to produce predictions.")
        
    print(f"Training completed. Val preds: {val_preds.shape}, Test preds: {test_preds.shape}")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "efficientnet_b0": train_efficientnet_b0,
}
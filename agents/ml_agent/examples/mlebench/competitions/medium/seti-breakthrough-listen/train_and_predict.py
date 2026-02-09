import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import timm
import numpy as np
from typing import Tuple, Dict, Any, Callable

# Task-adaptive type definitions
X = np.ndarray  # Preprocessed stacked spectrograms: (N, 1, 1638, 256)
y = np.ndarray  # Target labels: (N,)
Predictions = np.ndarray # Probability predictions: (N,)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Dataset Implementation =====

class SETIDataset(Dataset):
    """
    Custom Dataset for SETI cadence snippets.
    Supports returning indices to handle DistributedSampler reordering.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor = None):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        data = self.X[idx]
        if self.y is not None:
            return data, self.y[idx], idx
        return data, idx

# ===== Training Utilities =====

def mixup_data(x, y, alpha=0.4):
    """Applies Mixup augmentation to a batch of data."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Criterion for Mixup training."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===== Distributed Worker =====

def train_worker(rank: int, world_size: int, X_train: torch.Tensor, y_train: torch.Tensor, 
                 X_val: torch.Tensor, y_val: torch.Tensor, X_test: torch.Tensor, results_dict: Dict):
    """
    Worker function for DistributedDataParallel training.
    """
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Prepare Datasets and Samplers
    train_ds = SETIDataset(X_train, y_train.float())
    val_ds = SETIDataset(X_val)
    test_ds = SETIDataset(X_test)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    # H20-3e has massive VRAM (143GB), allowing for large batches even with B5
    train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, sampler=val_sampler, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, sampler=test_sampler, num_workers=8, pin_memory=True)
    
    # Build Model (EfficientNet-B5 with 1-channel input)
    model = timm.create_model('efficientnet_b5', pretrained=True, in_chans=1, num_classes=1)
    model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Configuration
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    epochs = 12
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs
    )
    
    scaler = GradScaler()
    
    # Training Loop
    print(f"[Rank {rank}] Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for data, target, _ in train_loader:
            data, target = data.cuda(rank, non_blocking=True), target.cuda(rank, non_blocking=True).unsqueeze(1)
            
            # Simple Augmentation: Horizontal Flip (Time axis)
            if np.random.rand() > 0.5:
                data = torch.flip(data, dims=[3])
            
            # Apply Mixup
            data, target_a, target_b, lam = mixup_data(data, target, alpha=0.4)
            
            optimizer.zero_grad()
            with autocast():
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
    # Inference Stage
    model.eval()
    
    def gather_predictions(loader, total_size):
        local_preds = []
        local_indices = []
        with torch.no_grad():
            for batch in loader:
                data, idx = batch
                data = data.cuda(rank, non_blocking=True)
                with autocast():
                    output = model(data).sigmoid()
                local_preds.append(output.cpu().numpy())
                local_indices.append(idx.numpy())
        
        # Concatenate local results
        preds_arr = np.concatenate(local_preds, axis=0)
        indices_arr = np.concatenate(local_indices, axis=0)
        
        # Gather results from all ranks
        world_preds = [None for _ in range(world_size)]
        world_indices = [None for _ in range(world_size)]
        dist.all_gather_object(world_preds, preds_arr)
        dist.all_gather_object(world_indices, indices_arr)
        
        if rank == 0:
            full_preds = np.concatenate(world_preds, axis=0).flatten()
            full_indices = np.concatenate(world_indices, axis=0).flatten()
            
            # Reconstruct original order using indices
            idx_to_pred = {}
            for i in range(len(full_indices)):
                idx_to_pred[full_indices[i]] = full_preds[i]
            
            return np.array([idx_to_pred[i] for i in range(total_size)])
        return None

    print(f"[Rank {rank}] Generating predictions...")
    v_preds = gather_predictions(val_loader, len(X_val))
    t_preds = gather_predictions(test_loader, len(X_test))
    
    if rank == 0:
        results_dict['val'] = v_preds
        results_dict['test'] = t_preds
        
    dist.destroy_process_group()

# ===== Main Training Function =====

def train_efficientnet_b5(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an EfficientNet-B5 model using Distributed Data Parallel (DDP) on 2 GPUs.
    Utilizes Mixup and OneCycleLR for robust training on astronomical spectrograms.
    """
    world_size = 2
    
    # Use torch shared memory to avoid serializing massive numpy arrays during mp.spawn
    X_train_t = torch.from_numpy(X_train).share_memory_()
    y_train_t = torch.from_numpy(y_train).share_memory_()
    X_val_t = torch.from_numpy(X_val).share_memory_()
    y_val_t = torch.from_numpy(y_val).share_memory_()
    X_test_t = torch.from_numpy(X_test).share_memory_()
    
    # Multiprocessing manager to retrieve results from workers
    manager = mp.Manager()
    results_dict = manager.dict()
    
    print(f"Initializing DDP on {world_size} GPUs for EfficientNet-B5 training...")
    
    try:
        mp.spawn(
            train_worker,
            args=(world_size, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, results_dict),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Distributed training failed: {e}")
        raise e
    
    # Retrieve and validate results
    val_preds = results_dict.get('val')
    test_preds = results_dict.get('test')
    
    if val_preds is None or test_preds is None:
        raise RuntimeError("Model training completed but failed to return predictions.")
        
    if np.isnan(val_preds).any() or np.isnan(test_preds).any():
        raise ValueError("Model predictions contain NaN values.")

    print(f"Training complete. Validation predictions shape: {val_preds.shape}")
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "efficientnet_b5": train_efficientnet_b5,
}
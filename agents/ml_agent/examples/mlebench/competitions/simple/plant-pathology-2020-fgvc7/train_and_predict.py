import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple, List, Dict, Any, Callable
import timm
import time
import copy

# Task-adaptive type definitions
X = Any           # PlantPathologyDataset
y = np.ndarray    # Target matrix (N, 4)
Predictions = np.ndarray # Probability matrix (N, 4)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class FocalLoss(nn.Module):
    """
    Focal Loss implementation as per specification.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy with label smoothing
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def get_model(model_name: str, num_classes: int = 4) -> nn.Module:
    """Creates a model from timm library."""
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def setup_ddp(rank: int, world_size: int):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def ddp_train_predict_worker(
    rank: int,
    world_size: int,
    model_name: str,
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    output_dict: Dict
):
    """
    Worker function for DDP training and prediction.
    """
    setup_ddp(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    
    # Hyperparameters
    epochs = 45
    batch_size = 32
    lr = 1e-4 * world_size # Scale LR by world size
    weight_decay = 1e-2
    
    # Data Loaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    # Model, Optimizer, Loss, Scheduler
    model = get_model(model_name).to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    
    # Scheduler: Linear Warmup + Cosine Annealing
    warmup_epochs = 5
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()
            
        # Validation on Rank 0
        if rank == 0:
            model.eval()
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
            val_loss = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.module.state_dict())
                
    # Synchronize and perform inference on Rank 0 using the best weights
    if rank == 0:
        model.module.load_state_dict(best_model_state)
        model.eval()
        
        def predict_with_tta(dataset):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            all_preds = []
            with torch.no_grad():
                for images in loader:
                    if isinstance(images, (list, tuple)): # Handle val_ds returning (img, label)
                        images = images[0]
                    images = images.to(device)
                    
                    # Original
                    logits1 = model(images)
                    # Horizontal Flip TTA
                    logits2 = model(torch.flip(images, dims=[3]))
                    
                    probs1 = F.softmax(logits1, dim=1)
                    probs2 = F.softmax(logits2, dim=1)
                    
                    avg_probs = (probs1 + probs2) / 2.0
                    all_preds.append(avg_probs.cpu().numpy())
            return np.concatenate(all_preds, axis=0)

        val_preds = predict_with_tta(val_ds)
        test_preds = predict_with_tta(test_ds)
        output_dict[model_name] = (val_preds, test_preds)
        
    cleanup_ddp()

def train_plant_pathology_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an ensemble of EfficientNetV2-S and ConvNeXt-Tiny using DDP.
    """
    print("Stage 4: Training Plant Pathology Ensemble (EfficientNetV2-S & ConvNeXt-Tiny)...")
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs available for training.")
    
    models_to_train = ['tf_efficientnetv2_s.in21k_ft_in1k', 'convnext_tiny.fb_in22k_ft_in1k']
    final_val_preds = []
    final_test_preds = []
    
    manager = mp.Manager()
    
    for model_name in models_to_train:
        print(f"Training model: {model_name} with DDP on {world_size} GPUs...")
        output_dict = manager.dict()
        
        mp.spawn(
            ddp_train_predict_worker,
            args=(world_size, model_name, X_train, X_val, X_test, output_dict),
            nprocs=world_size,
            join=True
        )
        
        val_p, test_p = output_dict[model_name]
        final_val_preds.append(val_p)
        final_test_preds.append(test_p)
        
        # Clear cache between models
        torch.cuda.empty_cache()

    # Averaging ensemble
    avg_val_preds = np.mean(final_val_preds, axis=0)
    avg_test_preds = np.mean(final_test_preds, axis=0)
    
    print(f"Training successful. Val shape: {avg_val_preds.shape}, Test shape: {avg_test_preds.shape}")
    
    return avg_val_preds, avg_test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "plant_pathology_ensemble": train_plant_pathology_ensemble,
}
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models
import numpy as np
from typing import Tuple, Any, Dict, Callable
import tempfile

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/rsna-miccai-brain-tumor-radiogenomic-classification/prepared/public"
OUTPUT_DATA_PATH = "output/dafb557f-655e-4395-9835-6f75549a5b27/1/executor/output"

# Task-adaptive type definitions
X = np.ndarray  # 5D NumPy array (N, 1, 64, 256, 256)
y = np.ndarray  # 1D NumPy array (N,)
Predictions = np.ndarray # 1D NumPy array (N,)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class BrainTumorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class EfficientNet3DModel(nn.Module):
    def __init__(self):
        super(EfficientNet3DModel, self).__init__()
        # Use ImageNet pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Depth, Height, Width)
        batch_size, channels, depth, height, width = x.shape
        
        # Reshape to process slices as a large batch: (Batch * Depth, Channels, Height, Width)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)
        
        # Convert 1-channel to 3-channels for EfficientNet
        x = x.repeat(1, 3, 1, 1)
        
        # Feature extraction
        features = self.backbone(x)  # (Batch * Depth, 1280)
        
        # Reshape back: (Batch, Depth, 1280)
        features = features.reshape(batch_size, depth, 1280)
        
        # Global Max Pooling across Depth dimension
        features = torch.max(features, dim=1)[0]  # (Batch, 1280)
        
        # Classification
        logits = self.classifier(features)
        return logits.squeeze(1)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, X_train, y_train, X_val, y_val, X_test, output_dict):
    setup(rank, world_size)
    
    # Hypers
    batch_size = 2 # 2 per GPU = 4 total
    epochs = 20
    lr = 1e-4
    patience = 5
    
    # Datasets & Loaders
    train_ds = BrainTumorDataset(X_train, y_train)
    val_ds = BrainTumorDataset(X_val, y_val)
    test_ds = BrainTumorDataset(X_test)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    # Model
    model = EfficientNet3DModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_path = os.path.join(tempfile.gettempdir(), f"best_model_fold.pth")
    
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # For validation, we calculate loss across all samples
            # Each rank processes a portion
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=4)
            
            for images, labels in val_loader:
                images, labels = images.to(rank), labels.to(rank)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        
        # Aggregate validation loss
        val_loss_tensor = torch.tensor([val_loss], device=rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / len(X_val)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.module.state_dict(), checkpoint_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        
        # Broadcast early stopping decision
        stop_signal = torch.tensor([1 if epochs_no_improve >= patience else 0], device=rank)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            if rank == 0: print("Early stopping triggered.")
            break

    # Final Inference
    dist.barrier()
    model.module.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    def predict(dataset):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        preds = []
        with torch.no_grad():
            for images in loader:
                images = images.to(rank) if isinstance(images, torch.Tensor) else images[0].to(rank)
                outputs = torch.sigmoid(model.module(images))
                preds.append(outputs.cpu().numpy())
        return np.concatenate(preds)

    # Only rank 0 performs full final prediction for simplicity since data is small
    if rank == 0:
        val_preds = predict(val_ds)
        test_preds = predict(test_ds)
        output_dict['val_preds'] = val_preds
        output_dict['test_preds'] = test_preds
        
    cleanup()

def train_efficientnet_b0_3d(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a 3D-aware EfficientNet-B0 model using DDP on multiple GPUs.
    """
    print(f"Training EfficientNet-B0 3D model on {torch.cuda.device_count()} GPUs...")
    
    manager = mp.Manager()
    output_dict = manager.dict()
    world_size = torch.cuda.device_count()
    
    # Use spawn to start DDP processes
    mp.spawn(
        train_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, output_dict),
        nprocs=world_size,
        join=True
    )
    
    val_preds = output_dict.get('val_preds')
    test_preds = output_dict.get('test_preds')
    
    if val_preds is None or test_preds is None:
        raise RuntimeError("Inference failed during DDP training.")

    print(f"Training complete. Val preds shape: {val_preds.shape}, Test preds shape: {test_preds.shape}")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "efficientnet_b0_3d": train_efficientnet_b0_3d,
}
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models
import numpy as np
from typing import Tuple, Any, Dict, Callable

# Task-adaptive type definitions
X = np.ndarray    # Preprocessed features (N, 128, 101)
y = np.ndarray    # Processed targets (N,)
Predictions = np.ndarray # Softmax probabilities (N, 31)

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/tensorflow-speech-recognition-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/fe927f25-451a-41da-a547-cdb392b784d8/1/executor/output"

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class AudioDataset(Dataset):
    """
    Custom Dataset for Log-Mel Spectrograms.
    Converts 1-channel spectrograms to 3-channels for transfer learning backbones.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Original shape (128, 101)
        x = self.X[idx]
        # Repeat across 3 channels to match EfficientNet/ResNet input expectations
        x_3ch = np.stack([x, x, x], axis=0)
        
        if self.y is not None:
            return torch.from_numpy(x_3ch).float(), torch.tensor(self.y[idx]).long()
        return torch.from_numpy(x_3ch).float()

class SpeechCNN(nn.Module):
    """
    EfficientNet-B0 based Classifier for 31-class Audio Recognition.
    """
    def __init__(self, num_classes: int = 31):
        super(SpeechCNN, self).__init__()
        # Using pre-trained weights for transfer learning
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Modify the classifier head for 31 classes
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def ddp_setup(rank: int, world_size: int):
    """
    Initializes the distributed process group.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_worker(rank: int, world_size: int, X_train, y_train, X_val, y_val, X_test, return_dict):
    """
    Worker function for Distributed Data Parallel (DDP) training.
    """
    ddp_setup(rank, world_size)
    
    # Create Datasets
    train_ds = AudioDataset(X_train, y_train)
    val_ds = AudioDataset(X_val, y_val)
    
    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    # Validation and Test loaders - only needed on rank 0 for final predictions or for all for val loss
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize Model, Loss, Optimizer and Scheduler
    model = SpeechCNN(num_classes=31).to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 25
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation Phase
        model.eval()
        local_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                local_val_loss += loss.item()
        
        # Aggregate validation loss across all GPUs
        val_loss_tensor = torch.tensor(local_val_loss).to(rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / (world_size * len(val_loader))
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Simple Early Stopping and Model Selection
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.module.state_dict(), "best_model_temp.pth")
            else:
                epochs_no_improve += 1
        
        # Synchronization for early stopping
        stop_signal = torch.tensor(1 if epochs_no_improve >= patience else 0).to(rank)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    # Final Inference on Rank 0
    if rank == 0:
        print("Training finished. Generating predictions on Rank 0...")
        model.module.load_state_dict(torch.load("best_model_temp.pth"))
        model.eval()
        
        def get_probs(data_array):
            ds = AudioDataset(data_array)
            loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)
            all_probs = []
            with torch.no_grad():
                for inputs in loader:
                    inputs = inputs.to(rank)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())
            return np.concatenate(all_probs, axis=0)

        val_preds = get_probs(X_val)
        test_preds = get_probs(X_test)
        
        return_dict['val_preds'] = val_preds
        return_dict['test_preds'] = test_preds
        
        if os.path.exists("best_model_temp.pth"):
            os.remove("best_model_temp.pth")

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
    Trains an EfficientNet-B0 model using Distributed Data Parallel across available GPUs.
    """
    print(f"Initializing DDP training for EfficientNet-B0. Input shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU available for training.")
    
    print(f"Using {world_size} GPUs for training.")
    
    # Use a Manager to retrieve results from the spawned processes
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Ensure X_train and others are accessible efficiently
    # mp.spawn uses 'fork' on Linux by default, so memory is shared efficiently
    mp.spawn(
        train_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, return_dict),
        nprocs=world_size,
        join=True
    )
    
    if 'val_preds' not in return_dict or 'test_preds' not in return_dict:
        raise RuntimeError("Training worker failed to return predictions.")
        
    val_preds = return_dict['val_preds']
    test_preds = return_dict['test_preds']
    
    print(f"Inference complete. Validation predictions shape: {val_preds.shape}, Test predictions shape: {test_preds.shape}")
    
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "efficientnet_b0": train_efficientnet_b0,
}
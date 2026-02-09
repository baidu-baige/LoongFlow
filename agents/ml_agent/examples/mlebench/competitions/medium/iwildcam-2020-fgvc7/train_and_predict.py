import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import pandas as pd
from typing import Tuple, Any, Dict, Callable
import time
import tempfile
import shutil

# Standard technical paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/iwildcam-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/9508c267-92c9-4fd0-91c8-90efc0fba263/1/executor/output"

# Task-adaptive type definitions
X = Any           # WildCamDataset from preprocess
y = pd.Series      # Target series
Predictions = np.ndarray # Probability arrays

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

def setup_ddp(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def ddp_worker(
    rank: int, 
    world_size: int, 
    X_train: X, 
    y_train: y, 
    X_val: X, 
    y_val: y, 
    X_test: X, 
    num_classes: int,
    class_weights: torch.Tensor,
    return_dict: Dict
):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 1. Build Model: ConvNeXt-Base
    # Note: IMAGENET1K_V1 is the weight set corresponding to pre-training on 21k and fine-tuning on 1k.
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # 2. Prepare DataLoaders with DistributedSampler
    train_sampler = DistributedSampler(X_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(X_val, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(X_test, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(X_train, batch_size=64, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(X_val, batch_size=64, sampler=val_sampler, num_workers=8, pin_memory=True)
    test_loader = DataLoader(X_test, batch_size=64, sampler=test_sampler, num_workers=8, pin_memory=True)

    # 3. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # Cosine Annealing with Warmup (5 epochs warmup, 20 epochs total)
    num_epochs = 20
    warmup_epochs = 5
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 4. Training Loop
    best_val_acc = 0.0
    patience = 5
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Sync validation accuracy
        stats = torch.tensor([correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        val_acc = (stats[0] / stats[1]).item()

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(OUTPUT_DATA_PATH, "best_model.pth"))
            else:
                no_improve += 1
        
        # Early Stopping broadcast
        stop_signal = torch.tensor(1.0 if no_improve >= patience else 0.0, device=device)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() > 0.5:
            break

    # 5. Inference (on Val and Test)
    dist.barrier()
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DATA_PATH, "best_model.pth"), map_location=device))
    model.eval()

    def get_probs(loader):
        probs_list = []
        with torch.no_grad():
            for imgs in loader:
                # Handle images or (images, labels)
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy())
        return np.concatenate(probs_list, axis=0)

    val_probs = get_probs(val_loader)
    test_probs = get_probs(test_loader)

    # Collect all predictions via shared return_dict
    return_dict[f"val_{rank}"] = val_probs
    return_dict[f"test_{rank}"] = test_probs

    cleanup_ddp()

def train_convnext_base(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a ConvNeXt-Base model using DistributedDataParallel on 2 GPUs.
    """
    print("Initializing ConvNeXt-Base training with DDP...")
    
    # Technical Parameters
    num_classes = 676  # Covers range 0-675 as per EDA
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Warning: Only {world_size} GPUs detected. Training might be slower.")
    
    # Calculate Class Weights
    counts = y_train.value_counts().reindex(range(num_classes), fill_value=0).values
    # Inverse square root weighting for extreme imbalance
    weights = np.where(counts > 0, 1.0 / np.sqrt(counts), 0)
    # Normalize weights to avoid extreme scaling
    if weights.sum() > 0:
        weights = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights)

    # Use multiprocessing Manager to collect results from ranks
    manager = mp.Manager()
    return_dict = manager.dict()

    # Launch DDP
    mp.spawn(
        ddp_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, num_classes, class_weights, return_dict),
        nprocs=world_size,
        join=True
    )

    print("Training complete. Merging predictions from all ranks...")

    # Merge results (DistributedSampler may reorder/pad, but we must reconstruct)
    # Because we use DistributedSampler, we need to be careful with indexing.
    # A simple but robust way is to store index in Dataset and sort, 
    # but here we assume the sampler's deterministic splitting.
    
    def reconstruct_preds(prefix, total_len):
        # DistributedSampler pads to make even batches across ranks.
        # It splits indices as: [0, 2, 4...] and [1, 3, 5...] (approx)
        all_indices = []
        all_probs = []
        
        for rank in range(world_size):
            probs = return_dict[f"{prefix}_{rank}"]
            # Reconstruct indices assigned to this rank by DistributedSampler
            rank_indices = list(range(rank, total_len + (world_size - (total_len % world_size)) if total_len % world_size != 0 else total_len, world_size))
            # Slice to remove padding
            valid_len = len([i for i in rank_indices if i < total_len])
            all_indices.extend([i for i in rank_indices if i < total_len])
            all_probs.extend(probs[:valid_len])
            
        # Re-sort to original order
        sorted_probs = [p for _, p in sorted(zip(all_indices, all_probs))]
        return np.array(sorted_probs)

    val_preds = reconstruct_preds("val", len(X_val))
    test_preds = reconstruct_preds("test", len(X_test))

    print(f"Final prediction shapes - Val: {val_preds.shape}, Test: {test_preds.shape}")
    
    # Clean up checkpoint
    best_model_path = os.path.join(OUTPUT_DATA_PATH, "best_model.pth")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "convnext_base": train_convnext_base,
}
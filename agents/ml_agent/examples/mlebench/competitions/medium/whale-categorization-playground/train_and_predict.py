import os
import math
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision
from typing import Tuple, Any, Dict, Callable

# ===== Type Definitions =====
X = Any           # PreprocessedWhaleDataset
y = Any           # torch.Tensor (Long)
Predictions = Any # np.ndarray of scores

# Model Function Type as per specification
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Model Components =====

class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) head.
    Optimized to maximize the angular margin between 4,000+ whale classes.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        # L2 normalize input and weights for cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if label is None:
            # Inference mode: return scaled cosine similarity for prediction/embedding
            return cosine * self.s
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        # Handling the boundary condition for theta + margin > pi
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class ConvNextArcFaceModel(nn.Module):
    """
    ConvNeXt-Base backbone with ArcFace classification head.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ConvNextArcFaceModel, self).__init__()
        
        # Load backbone with error handling for offline environments
        weights = 'IMAGENET1K_V1' if pretrained else None
        try:
            self.backbone = torchvision.models.convnext_base(weights=weights)
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights ({e}). Initializing with weights=None.")
            self.backbone = torchvision.models.convnext_base(weights=None)
        
        # Extract feature dimension and strip original classifier head
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0], # LayerNorm
            self.backbone.classifier[1]  # Flatten
        )
        self.arc_face = ArcMarginProduct(in_features, num_classes, s=30.0, m=0.50)
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        features = self.backbone(x)
        return self.arc_face(features, labels)

class IndexedDataset(Dataset):
    """
    Wrapper to track original indices during distributed inference for perfect alignment.
    """
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        # PreprocessedWhaleDataset returns img or (img, label)
        if isinstance(data, (tuple, list)):
            return data + (idx,)
        return data, idx

# ===== Distributed Training Logic =====

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def ddp_worker(rank: int, world_size: int, port: int, X_train: X, X_val: X, X_test: X, num_classes: int, return_dict: Dict):
    """
    Distributed Data Parallel worker process.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Model definition - Try to load pretrained once per rank (ideally Rank 0 first but simple try-catch is robust)
    model = ConvNextArcFaceModel(num_classes, pretrained=True).to(device)
    model = DDP(model, device_ids=[rank])

    # Samplers and Loaders
    # X_train is already wrapped in PreprocessedWhaleDataset from upstream
    train_sampler = DistributedSampler(X_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_ds_indexed = IndexedDataset(X_val)
    test_ds_indexed = IndexedDataset(X_test)
    
    val_sampler = DistributedSampler(val_ds_indexed, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds_indexed, num_replicas=world_size, rank=rank, shuffle=False)

    # Batch size: 8 per GPU (Total 16) for 384x768 resolution
    train_loader = DataLoader(X_train, batch_size=8, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds_indexed, batch_size=16, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds_indexed, batch_size=16, sampler=test_sampler, num_workers=4, pin_memory=True)

    # Optimization Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    num_epochs = 30
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs, labels)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    # Inference logic
    model.eval()
    def predict(loader):
        preds_list, idx_list = [], []
        with torch.no_grad():
            for batch in loader:
                # batch is (imgs, labels, idxs) or (imgs, idxs)
                imgs = batch[0].to(device)
                idxs = batch[-1]
                with autocast():
                    outputs = model(imgs) # ArcFace head returns scaled similarity
                preds_list.append(outputs.cpu().numpy())
                idx_list.append(idxs.numpy())
        return np.concatenate(preds_list, axis=0), np.concatenate(idx_list, axis=0)

    val_p, val_i = predict(val_loader)
    test_p, test_i = predict(test_loader)

    # Gather results from all ranks to reconstruct original order
    all_val_p = [None] * world_size
    all_val_i = [None] * world_size
    all_test_p = [None] * world_size
    all_test_i = [None] * world_size

    dist.all_gather_object(all_val_p, val_p)
    dist.all_gather_object(all_val_i, val_i)
    dist.all_gather_object(all_test_p, test_p)
    dist.all_gather_object(all_test_i, test_i)

    if rank == 0:
        def assemble(preds_list, idx_list):
            p = np.concatenate(preds_list, axis=0)
            i = np.concatenate(idx_list, axis=0)
            # Remove DDP padding and sort by original index
            unique_idx, first_indices = np.unique(i, return_index=True)
            return p[first_indices][np.argsort(unique_idx)]

        return_dict['val_preds'] = assemble(all_val_p, all_val_i)
        return_dict['test_preds'] = assemble(all_test_p, all_test_i)

    dist.destroy_process_group()

# ===== Training Functions =====

def train_convnext_arcface(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains ConvNeXt-Base with ArcFace using Distributed Data Parallel on 2 GPUs.
    Handles class imbalance and fine-grained identification via angular margin loss.
    """
    print("Initializing training for ConvNeXt-ArcFace...")
    # Determine class count from processed labels
    num_classes = int(y_train.max().item() + 1)
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU available for training.")
    
    port = find_free_port()
    
    # Use Manager for inter-process communication of results
    manager = mp.Manager()
    return_dict = manager.dict()

    # Attempt to trigger download/caching in main process first (to avoid DDP race conditions)
    try:
        _ = torchvision.models.convnext_base(weights='IMAGENET1K_V1')
    except Exception:
        pass

    print(f"Spawning DDP processes across {world_size} GPUs...")
    mp.spawn(
        ddp_worker,
        args=(world_size, port, X_train, X_val, X_test, num_classes, return_dict),
        nprocs=world_size,
        join=True
    )

    val_preds = return_dict.get('val_preds')
    test_preds = return_dict.get('test_preds')

    if val_preds is None or test_preds is None:
        raise RuntimeError("Distributed training failed or returned empty results.")

    print(f"Training Stage Successful. Generated val_preds shape: {val_preds.shape}, test_preds shape: {test_preds.shape}")
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "convnext_arcface": train_convnext_arcface,
}
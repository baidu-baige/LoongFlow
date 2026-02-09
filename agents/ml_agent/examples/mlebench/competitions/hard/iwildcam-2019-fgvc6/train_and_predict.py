import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import numpy as np
import pandas as pd
import os
from typing import Tuple, Any, Dict, Callable

# Paths defined in the environment
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-09/evolux/output/mlebench/iwildcam-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/939424a9-5a07-4c99-9f56-709187b5b05c/1/executor/output"

# Task-adaptive type definitions
X = Any           # IWildCamDataset object (emits Image, Meta)
y = pd.Series     # Target vector (category_id)
Predictions = np.ndarray # Probability matrix

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# Global Configuration
NUM_CLASSES = 23
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.05

class MetaFusedConvNeXt(nn.Module):
    """
    Backbone: convnext_base (pretrained on ImageNet)
    Meta-Head: 2-layer MLP (128-64 units) for metadata fused with CNN pooling output.
    """
    def __init__(self, num_classes: int = 23):
        super().__init__()
        # Load backbone with pretrained ImageNet weights
        self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        
        # ConvNeXt-Base features are 1024-dim
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        
        # Meta MLP Head for metadata (seq_num_frames, frame_num)
        self.meta_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # Fusion Classifier
        self.classifier = nn.Linear(in_features + 64, num_classes)
        
    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        visual_feat = self.backbone(img) # (B, 1024)
        meta_feat = self.meta_mlp(meta)   # (B, 64)
        combined = torch.cat([visual_feat, meta_feat], dim=1) # (B, 1088)
        return self.classifier(combined)

class LabelWrapper(Dataset):
    """
    Attaches labels to the preprocessed dataset for training.
    """
    def __init__(self, dataset: Any, labels: pd.Series = None):
        self.dataset = dataset
        self.labels = labels.values if labels is not None else None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, meta = self.dataset[idx]
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, meta, label
        return img, meta

def ddp_worker(rank: int, world_size: int, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, class_weights: torch.Tensor, shared_results: Dict):
    """
    Worker function for DDP training and prediction.
    """
    # Environment for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Model Setup
    model = MetaFusedConvNeXt(NUM_CLASSES).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(rank))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # DataLoaders with DistributedSampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=8, pin_memory=True)
    
    # Training Loop
    if rank == 0: print(f"Starting training on {world_size} GPUs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        for imgs, metas, labels in train_loader:
            imgs, metas, labels = imgs.to(rank), metas.to(rank), labels.to(rank)
            optimizer.zero_grad()
            logits = model(imgs, metas)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        if rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Loss: {total_loss/len(train_loader):.4f}")

    # Inference logic
    def run_inference(dataset: Dataset) -> np.ndarray:
        model.eval()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=8, pin_memory=True)
        
        # Track local results and indices to reorder later
        indices_list = list(range(len(dataset)))
        total_size = (len(dataset) + world_size - 1) // world_size * world_size
        indices_list += indices_list[:(total_size - len(indices_list))]
        rank_indices = indices_list[rank:total_size:world_size]
        
        local_probs = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                imgs, metas = batch[0].to(rank), batch[1].to(rank)
                logits = model(imgs, metas)
                local_probs.append(torch.softmax(logits, dim=1))
        
        local_probs_tensor = torch.cat(local_probs, dim=0)
        local_indices_tensor = torch.tensor(rank_indices, device=rank, dtype=torch.long)
        
        # Gather all
        gathered_probs = [torch.zeros_like(local_probs_tensor) for _ in range(world_size)]
        gathered_indices = [torch.zeros_like(local_indices_tensor) for _ in range(world_size)]
        
        torch.distributed.all_gather(gathered_probs, local_probs_tensor)
        torch.distributed.all_gather(gathered_indices, local_indices_tensor)
        
        if rank == 0:
            all_probs = torch.cat(gathered_probs, dim=0).cpu().numpy()
            all_idxs = torch.cat(gathered_indices, dim=0).cpu().numpy()
            
            final_probs = np.zeros((len(dataset), NUM_CLASSES))
            for p, idx in zip(all_probs, all_idxs):
                if idx < len(dataset):
                    final_probs[idx] = p
            return final_probs
        return None

    # Get predictions
    val_p = run_inference(val_ds)
    test_p = run_inference(test_ds)
    
    if rank == 0:
        shared_results['val'] = val_p
        shared_results['test'] = test_p

    torch.distributed.destroy_process_group()

def train_convnext_base_meta(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains ConvNeXt-Base with Metadata fusion using DDP.
    """
    # 1. Calc Class Weights
    counts = y_train.value_counts().sort_index()
    weights = torch.ones(NUM_CLASSES)
    for cls_id, count in counts.items():
        if cls_id < NUM_CLASSES:
            weights[cls_id] = 1.0 / np.log(count + 1.1)
    
    # 2. Prepare Datasets
    train_ds = LabelWrapper(X_train, y_train)
    val_ds = LabelWrapper(X_val)
    test_ds = LabelWrapper(X_test)
    
    # 3. Spawn DDP Processes
    world_size = torch.cuda.device_count()
    manager = mp.Manager()
    shared_results = manager.dict()
    
    mp.spawn(
        ddp_worker,
        nprocs=world_size,
        args=(world_size, train_ds, val_ds, test_ds, weights, shared_results),
        join=True
    )
    
    val_preds = shared_results.get('val')
    test_preds = shared_results.get('test')
    
    if val_preds is None or test_preds is None:
        raise RuntimeError("DDP training failed to produce predictions.")
        
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "convnext_base_meta": train_convnext_base_meta,
}
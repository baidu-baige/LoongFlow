import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.transforms import v2
import pandas as pd
import numpy as np
import os
import timm
from typing import Tuple, Any, Dict, Callable, List

# Task-adaptive type definitions
X = Any           # Feature matrix type (ModelReadyLoader)
y = Any           # Target vector type (pd.Series)
Predictions = Any # Model predictions type (np.ndarray)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class IndexedDataset(Dataset):
    """Wrapper to return index with data for DDP alignment."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx], idx

class TrainCollate:
    """Picklable collation class for Mixup/CutMix in DDP training."""
    def __init__(self):
        self.mixup = v2.MixUp(alpha=0.2, num_classes=2)
        self.cutmix = v2.CutMix(alpha=1.0, num_classes=2)
        self.cutmix_or_mixup = v2.RandomChoice([self.mixup, self.cutmix])

    def __call__(self, batch):
        data_list, indices = zip(*batch)
        # Apply Mixup/CutMix to the batched tensors
        collated_data = self.cutmix_or_mixup(*torch.utils.data.default_collate(list(data_list)))
        return collated_data, torch.tensor(indices)

def eval_collate(batch):
    """Picklable collation function for evaluation."""
    data_list, indices = zip(*batch)
    collated_data = torch.utils.data.default_collate(list(data_list))
    return collated_data, torch.tensor(indices)

def ddp_setup(rank: int, world_size: int):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def _train_worker(
    rank: int, 
    world_size: int, 
    train_ds: Dataset, 
    val_ds: Dataset, 
    test_ds: Dataset, 
    results_dict: Dict
):
    """Worker function for DDP training using High-Capacity Vision Ensembles."""
    ddp_setup(rank, world_size)
    
    # Model Configurations
    # We use high-capacity models compatible with the 384x384 resolution provided by preprocess.
    # Replaced eva02 with vit_large_patch16_384 which is more likely to be available in standard timm.
    model_configs = [
        {"name": "convnext_large.fb_in22k_ft_in1k", "lr": 3e-5},
        {"name": "swin_large_patch4_window12_384.ms_in22k_ft_in1k", "lr": 2e-5},
        {"name": "vit_large_patch16_384.augreg_in21k_ft_in1k", "lr": 2e-5}
    ]
    
    # Batch size selected to utilize H20-3e 140GB VRAM efficiently.
    batch_size = 32 # Per GPU
    num_epochs = 6
    
    # Create Indexed Datasets for alignment
    train_idx_ds = IndexedDataset(train_ds)
    val_idx_ds = IndexedDataset(val_ds)
    test_idx_ds = IndexedDataset(test_ds)
    
    # Samplers for DDP
    train_sampler = DistributedSampler(train_idx_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_idx_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_idx_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    # DataLoaders
    train_loader = DataLoader(
        train_idx_ds, batch_size=batch_size, sampler=train_sampler, 
        collate_fn=TrainCollate(), num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_idx_ds, batch_size=batch_size, sampler=val_sampler, 
        collate_fn=eval_collate, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_idx_ds, batch_size=batch_size, sampler=test_sampler, 
        collate_fn=eval_collate, num_workers=4, pin_memory=True
    )

    all_val_preds_ensemble = []
    all_test_preds_ensemble = []

    for cfg in model_configs:
        if rank == 0: print(f"Processing model: {cfg['name']}")
        
        # Build Model
        try:
            model = timm.create_model(cfg['name'], pretrained=True, num_classes=2).to(rank)
        except Exception as e:
            if rank == 0: print(f"Model {cfg['name']} not found, attempting generic variant...")
            base_name = cfg['name'].split('.')[0]
            model = timm.create_model(base_name, pretrained=True, num_classes=2).to(rank)

        model = DDP(model, device_ids=[rank])
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.05)
        
        # Scheduler: Warmup + Cosine
        warmup_sch = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
        cosine_sch = CosineAnnealingLR(optimizer, T_max=num_epochs - 3)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[3])
        
        # Training Loop
        for epoch in range(num_epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            for (images, targets), _ in train_loader:
                images, targets = images.to(rank), targets.to(rank)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        # Inference with TTA (2-crop Horizontal Flip)
        def predict_with_tta(loader, dataset_len):
            model.eval()
            local_preds, local_idxs = [], []
            with torch.no_grad():
                for data, idx in loader:
                    img = data[0] if isinstance(data, (list, tuple)) else data
                    img = img.to(rank)
                    
                    # TTA: Avg probabilities of original and flipped
                    out = torch.softmax(model(img), dim=1)
                    out_flip = torch.softmax(model(torch.flip(img, dims=[-1])), dim=1)
                    
                    avg_prob = (out + out_flip) / 2
                    local_preds.append(avg_prob[:, 1]) # Prob(Dog)
                    local_idxs.append(idx.to(rank))
            
            if not local_preds: return np.array([])

            local_preds_cat = torch.cat(local_preds)
            local_idxs_cat = torch.cat(local_idxs)
            
            # Gather results from all GPUs
            gathered_preds = [torch.zeros_like(local_preds_cat) for _ in range(world_size)]
            gathered_idxs = [torch.zeros_like(local_idxs_cat) for _ in range(world_size)]
            
            dist.all_gather(gathered_preds, local_preds_cat)
            dist.all_gather(gathered_idxs, local_idxs_cat)
            
            full_preds = torch.cat(gathered_preds).cpu().numpy()
            full_idxs = torch.cat(gathered_idxs).cpu().numpy()
            
            # Re-sort to original order and remove DDP padding
            sort_idx = np.argsort(full_idxs)
            full_preds = full_preds[sort_idx]
            full_idxs = full_idxs[sort_idx]
            
            _, unique_indices = np.unique(full_idxs, return_index=True)
            return full_preds[unique_indices][:dataset_len]

        val_preds = predict_with_tta(val_loader, len(val_ds))
        test_preds = predict_with_tta(test_loader, len(test_ds))
        
        all_val_preds_ensemble.append(val_preds)
        all_test_preds_ensemble.append(test_preds)
        
        del model
        torch.cuda.empty_cache()

    if rank == 0:
        results_dict['val'] = np.mean(all_val_preds_ensemble, axis=0)
        results_dict['test'] = np.mean(all_test_preds_ensemble, axis=0)
    
    ddp_cleanup()

def train_vision_ensemble_ddp(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a high-capacity ensemble of vision models (ConvNeXt-L, Swin-L, ViT-L) using DDP.
    Returns OOF and Test predictions.
    """
    print("Initiating Training: Vision Ensemble with DDP Acceleration...")
    
    # Upstream data structure contains ModelReadyLoaders
    train_ds = X_train.dataset
    val_ds = X_val.dataset
    test_ds = X_test.dataset
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs available for vision ensemble training.")

    manager = mp.Manager()
    results_dict = manager.dict()
    
    # Spawn distributed processes
    mp.spawn(
        _train_worker,
        args=(world_size, train_ds, val_ds, test_ds, results_dict),
        nprocs=world_size,
        join=True
    )
    
    val_preds = results_dict.get('val')
    test_preds = results_dict.get('test')
    
    if val_preds is None or test_preds is None:
        raise RuntimeError("Distributed training failed to generate valid predictions.")
        
    print(f"Training Complete. OOF Predictions generated for {len(val_preds)} samples.")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "vision_ensemble_ddp": train_vision_ensemble_ddp,
}
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
import timm
from typing import Tuple, Any, Dict, Callable, List
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/histopathologic-cancer-detection/prepared/public"
OUTPUT_DATA_PATH = "output/cf83edc4-8764-4cf8-95a0-4f4a823260c7/2/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Dataset Implementation =====

class CancerDataset(Dataset):
    def __init__(self, images: List[np.ndarray], labels: np.ndarray = None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.labels is not None:
            return image, torch.tensor(self.labels[idx], dtype=torch.float32), idx
        return image, idx

# ===== Model Implementation =====

class DualBackboneModel(nn.Module):
    def __init__(self, model_names: List[str] = ["convnext_small.fb_in22k_ft_in1k", "swin_small_patch4_window7_224"]):
        super().__init__()
        # Pretrained models from timm. These are high-capacity and support 224x224 input.
        self.backbone1 = timm.create_model(model_names[0], pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model(model_names[1], pretrained=True, num_classes=0)
        
        # Verify feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            f1 = self.backbone1(dummy).shape[1]
            f2 = self.backbone2(dummy).shape[1]
        
        self.head = nn.Sequential(
            nn.Linear(f1 + f2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Center patches are 96x96, interpolate to 224x224 for standard backbone input
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        feat1 = self.backbone1(x)
        feat2 = self.backbone2(x)
        combined = torch.cat([feat1, feat2], dim=1)
        return self.head(combined)

# ===== Training Worker =====

def ddp_worker(rank: int, world_size: int, data_dict: Dict, return_dict: Dict):
    """
    Worker function for Distributed Data Parallel training.
    """
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Datasets
        train_ds = CancerDataset(data_dict['X_train_imgs'], data_dict['y_train_vals'], transform=data_dict['train_transform'])
        val_ds = CancerDataset(data_dict['X_val_imgs'], data_dict['y_val_vals'], transform=data_dict['val_transform'])
        test_ds = CancerDataset(data_dict['X_test_imgs'], transform=data_dict['val_transform'])

        # Samplers & Loaders
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

        batch_size = 48
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

        model = DualBackboneModel().to(device)
        model = DDP(model, device_ids=[rank])

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        num_epochs = 4
        total_steps = len(train_loader) * num_epochs
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-4, total_steps=total_steps, 
            pct_start=0.1, anneal_strategy='cos'
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Phase
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
            dist.barrier()

        # Inference Phase
        def get_predictions(loader, tta_list=None):
            model.eval()
            active_ttas = tta_list if tta_list else [data_dict['val_transform']]
            accum_p = None
            final_i = None
            
            with torch.no_grad():
                for t_idx, tta_tf in enumerate(active_ttas):
                    loader.dataset.transform = tta_tf
                    batch_probs = []
                    batch_indices = []
                    for batch_data in loader:
                        # Handle potential label in batch_data
                        if len(batch_data) == 3:
                            images, _, idxs = batch_data
                        else:
                            images, idxs = batch_data
                            
                        images = images.to(device)
                        outputs = torch.sigmoid(model(images))
                        batch_probs.append(outputs.cpu().numpy())
                        batch_indices.append(idxs.cpu().numpy())
                    
                    tta_p = np.concatenate(batch_probs)
                    tta_i = np.concatenate(batch_indices)
                    
                    if t_idx == 0:
                        accum_p = tta_p
                        final_i = tta_i
                    else:
                        accum_p += tta_p
                
                accum_p /= len(active_ttas)
            
            # Gather results from all GPUs
            gathered_p = [None for _ in range(world_size)]
            gathered_i = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_p, accum_p)
            dist.all_gather_object(gathered_i, final_i)
            
            if rank == 0:
                full_p = np.concatenate(gathered_p)
                full_i = np.concatenate(gathered_i)
                # Re-sort and filter out DistributedSampler padding duplicates
                unique_idx, first_idx = np.unique(full_i, return_index=True)
                valid_mask = unique_idx < len(loader.dataset)
                sorted_p = full_p[first_idx][np.argsort(unique_idx[valid_mask])]
                return sorted_p.flatten()
            return None

        val_res = get_predictions(val_loader)
        test_res = get_predictions(test_loader, tta_list=data_dict['tta_transforms'])

        if rank == 0:
            return_dict['val_preds'] = val_res
            return_dict['test_preds'] = test_res

    finally:
        dist.destroy_process_group()

# ===== Main Training Function =====

def train_dual_backbone(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a Dual Backbone (ConvNeXt + Swin) ensemble using DDP across 4 GPUs.
    Handles D4 symmetry via TTA and optimizes for AUC-ROC.
    """
    print(f"Initializing Dual Backbone Ensemble Training (ConvNeXt-S + Swin-S)...")
    
    # Prepare data dictionary for workers (avoiding pickling overhead in loops)
    data_dict = {
        'X_train_imgs': list(X_train['image'].values),
        'y_train_vals': y_train.values.astype(np.float32),
        'train_transform': X_train.attrs['transform'],
        'X_val_imgs': list(X_val['image'].values),
        'y_val_vals': y_val.values.astype(np.float32),
        'val_transform': X_val.attrs['transform'],
        'X_test_imgs': list(X_test['image'].values),
        'tta_transforms': X_test.attrs['tta_transforms']
    }

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU available for DDP training.")
    
    manager = mp.Manager()
    return_dict = manager.dict()
    
    print(f"Executing DDP Training across {world_size} GPUs...")
    mp.spawn(ddp_worker, args=(world_size, data_dict, return_dict), nprocs=world_size, join=True)
    
    val_preds = return_dict.get('val_preds')
    test_preds = return_dict.get('test_preds')
    
    if val_preds is None or test_preds is None:
        raise RuntimeError("Training process failed to return valid predictions.")
        
    print("Dual Backbone training and inference completed successfully.")
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "dual_backbone_ensemble": train_dual_backbone,
}
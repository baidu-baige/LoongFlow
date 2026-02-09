import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Any, Dict, Callable
import timm

# Task-adaptive type definitions
X = torch.Tensor  # Preprocessed image tensors (N, 3, 512, 512)
y = torch.Tensor  # Target labels (N,)
Predictions = Dict[str, np.ndarray] # Dictionary mapping model name to probability matrix

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Bi-Tempered Logistic Loss Implementation =====

def log_t(u, t):
    if t == 1.0:
        return torch.log(u)
    else:
        return (u**(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u)**(1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters):
    mu = torch.max(activations, dim=-1, keepdim=True).values
    normalized_activations_step_0 = activations - mu
    normalized_activations = normalized_activations_step_0
    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1, keepdim=True)
        normalized_activations = normalized_activations_step_0 - log_t(logt_partition, t)
    return exp_t(normalized_activations, t)

def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.1, num_iters=5):
    num_classes = activations.shape[-1]
    if labels.dim() == 1:
        labels = F.one_hot(labels, num_classes).float()
    
    if label_smoothing > 0:
        labels = labels * (1.0 - label_smoothing) + label_smoothing / num_classes

    probabilities = compute_normalization_fixed_point(activations, t2, num_iters)
    temp_term = (labels * log_t(labels + 1e-10, t1) - labels * log_t(probabilities + 1e-10, t1))
    loss = torch.sum(temp_term, dim=-1) - torch.sum((labels**(2.0 - t1) - probabilities**(2.0 - t1)) / (2.0 - t1), dim=-1)
    return loss.mean()

# ===== Regularization: Mixup & CutMix =====

def get_mixup_cutmix_data(x, y, alpha_mixup=0.4, alpha_cutmix=1.0):
    if np.random.rand() > 0.5:
        # Mixup
        lam = np.random.beta(alpha_mixup, alpha_mixup)
        index = torch.randperm(x.size(0)).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    else:
        # CutMix
        lam = np.random.beta(alpha_cutmix, alpha_cutmix)
        index = torch.randperm(x.size(0)).cuda()
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_cloned = x.clone()
        x_cloned[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x_cloned, y_a, y_b, lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# ===== Dataset =====

class CassavaDataset(Dataset):
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.images[idx], self.labels[idx]
        return self.images[idx]

# ===== Training & Inference Worker =====

def ddp_worker(rank, world_size, X_train, y_train, X_val, y_val, X_test, return_dict):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Specification: ConvNeXt-Base and Swin-Base at 384 resolution
    models_config = [
        ("convnext", "convnext_base.fb_in22k_ft_in1k", 16, 4e-5),
        ("swin", "swin_base_patch4_window12_384.ms_in22k_ft_in1k", 8, 2e-5)
    ]
    
    IMG_SIZE = 384
    EPOCHS = 10

    for model_key, model_name, bs, lr in models_config:
        if rank == 0:
            print(f"Rank 0: Starting training {model_name} for {EPOCHS} epochs...")
            
        model = timm.create_model(model_name, pretrained=True, num_classes=5).cuda()
        model = DDP(model, device_ids=[rank])
        
        train_ds = CassavaDataset(X_train, y_train)
        val_ds = CassavaDataset(X_val, y_val)
        test_ds = CassavaDataset(X_test)
        
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs*2, sampler=val_sampler, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=bs*2, sampler=test_sampler, num_workers=8, pin_memory=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler()
        
        for epoch in range(EPOCHS):
            model.train()
            train_sampler.set_epoch(epoch)
            for imgs, labels in train_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                imgs_mixed, y_a, y_b, lam = get_mixup_cutmix_data(imgs, labels)
                
                # Resize from 512 to 384
                imgs_mixed = F.interpolate(imgs_mixed, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
                
                optimizer.zero_grad()
                with autocast():
                    outputs = model(imgs_mixed)
                    loss = lam * bi_tempered_logistic_loss(outputs, y_a, 0.8, 1.2) + \
                           (1 - lam) * bi_tempered_logistic_loss(outputs, y_b, 0.8, 1.2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

        # Inference with 4-way TTA
        def predict_tta(loader, dataset_len):
            model.eval()
            local_preds = []
            with torch.no_grad():
                for batch in loader:
                    imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    imgs = imgs.cuda()
                    imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
                    with autocast():
                        # Standard
                        p1 = model(imgs).softmax(1)
                        # Horizontal Flip
                        p2 = model(torch.flip(imgs, dims=[3])).softmax(1)
                        # Vertical Flip
                        p3 = model(torch.flip(imgs, dims=[2])).softmax(1)
                        # Both Flips
                        p4 = model(torch.flip(imgs, dims=[2, 3])).softmax(1)
                        p_avg = (p1 + p2 + p3 + p4) / 4.0
                        local_preds.append(p_avg)
            
            local_preds = torch.cat(local_preds, dim=0)
            # Gather all predictions from all GPUs
            # DistributedSampler pads to make it divisible by world_size
            world_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
            dist.all_gather(world_preds, local_preds)
            
            # Reconstruction: sampler with shuffle=False uses rank::world_size indices
            # Interleave results to get original order
            full_preds = torch.stack(world_preds, dim=1).view(-1, 5)
            # Truncate padding added by DistributedSampler
            return full_preds[:dataset_len].cpu().numpy()

        val_probs = predict_tta(val_loader, len(val_ds))
        test_probs = predict_tta(test_loader, len(test_ds))
        
        if rank == 0:
            return_dict[f'{model_key}_val'] = val_probs
            return_dict[f'{model_key}_test'] = test_probs
            
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        dist.barrier()

    dist.destroy_process_group()

# ===== Main Training Function =====

def train_high_res_vision_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains ConvNeXt-Base and Swin-Base at 384px resolution using DDP and AMP.
    Returns decoupled predictions for downstream weighted ensembling.
    """
    print("Initializing High-Res Vision Ensemble training (ConvNeXt & Swin)...")
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(f"Required 2 GPUs for DDP, but found {world_size}")
        
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Use mp.spawn for clean DDP orchestration
    mp.spawn(
        ddp_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, return_dict),
        nprocs=world_size,
        join=True
    )
    
    # Extract decoupled model predictions
    val_preds = {
        "convnext": return_dict.get("convnext_val"),
        "swin": return_dict.get("swin_val")
    }
    test_preds = {
        "convnext": return_dict.get("convnext_test"),
        "swin": return_dict.get("swin_test")
    }
    
    # Sanity checks
    for key in ["convnext", "swin"]:
        if val_preds[key] is None or test_preds[key] is None:
            raise RuntimeError(f"Training failed for model: {key}")
        if np.isnan(val_preds[key]).any() or np.isnan(test_preds[key]).any():
            raise ValueError(f"NaN detected in predictions for model: {key}")

    print("Training complete. Decoupled predictions generated.")
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "high_res_vision_ensemble": train_high_res_vision_ensemble,
}
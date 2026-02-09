import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, transforms
import pandas as pd
import numpy as np
from typing import Tuple, Any, Dict, Callable
from PIL import Image

# Pipeline configuration constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/herbarium-2020-fgvc7/prepared/public"
OUTPUT_DATA_PATH = "output/cf84c6d3-8647-45ca-b9ff-02e7ed67cf5b/1/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Feature matrix containing file paths and metadata
y = torch.Tensor      # Target vector (category indices)
Predictions = np.ndarray # Model predictions

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Model Components =====

class HerbariumDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: torch.Tensor = None, transform=None):
        self.file_names = X['file_name'].values
        self.genus_ids = X['genus_id'].values if 'genus_id' in X.columns else None
        self.family_ids = X['family_id'].values if 'family_id' in X.columns else None
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (448, 448), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        if self.y is not None:
            target = {
                'species': self.y[idx],
                'genus': torch.tensor(self.genus_ids[idx], dtype=torch.long),
                'family': torch.tensor(self.family_ids[idx], dtype=torch.long)
            }
            return image, target
        
        return image

class MultiTaskConvNeXt(nn.Module):
    def __init__(self, num_species, num_genus, num_family):
        super(MultiTaskConvNeXt, self).__init__()
        self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[2].in_features
        # Replace the classifier to remove the final linear layer but keep the backbone structure
        self.backbone.classifier = nn.Identity()
        
        self.head_species = nn.Linear(in_features, num_species)
        self.head_genus = nn.Linear(in_features, num_genus)
        self.head_family = nn.Linear(in_features, num_family)

    def forward(self, x):
        features = self.backbone(x)
        # ConvNeXt backbone with Identity classifier returns (N, 1024, 1, 1) after avgpool
        features = torch.flatten(features, 1)
        out_species = self.head_species(features)
        out_genus = self.head_genus(features)
        out_family = self.head_family(features)
        return out_species, out_genus, out_family

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, sample_per_class):
        super(BalancedSoftmaxLoss, self).__init__()
        counts = torch.tensor(sample_per_class).float()
        counts[counts == 0] = 1.0
        self.register_buffer('log_prior', torch.log(counts))

    def forward(self, logits, labels):
        # Balanced Softmax: CrossEntropy(logits + log(n_i), labels)
        adjusted_logits = logits + self.log_prior
        return F.cross_entropy(adjusted_logits, labels)

# ===== Distributed Worker =====

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, X_train, y_train, X_val, y_val, X_test, return_dict):
    setup(rank, world_size)
    
    # Constants from Technical Spec and EDA
    num_species, num_genus, num_family = 32093, 3678, 310
    batch_size = 32
    lr = 1e-4
    weight_decay = 0.05
    epochs = 4 # Balanced for time and quality
    grad_accum_steps = 2
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets and Samplers
    train_ds = HerbariumDataset(X_train, y_train, transform=train_transform)
    val_ds = HerbariumDataset(X_val, y_val, transform=val_transform)
    test_ds = HerbariumDataset(X_test, transform=val_transform)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    
    # Model Setup
    model = MultiTaskConvNeXt(num_species, num_genus, num_family).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Loss Functions
    class_counts = np.zeros(num_species)
    unique, counts = np.unique(y_train.numpy(), return_counts=True)
    class_counts[unique.astype(int)] = counts
    criterion_species = BalancedSoftmaxLoss(class_counts).to(rank)
    criterion_genus = nn.CrossEntropyLoss().to(rank)
    criterion_family = nn.CrossEntropyLoss().to(rank)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(rank)
            t_species = targets['species'].to(rank)
            t_genus = targets['genus'].to(rank)
            t_family = targets['family'].to(rank)
            
            with torch.cuda.amp.autocast():
                p_species, p_genus, p_family = model(images)
                loss_species = criterion_species(p_species, t_species)
                loss_genus = criterion_genus(p_genus, t_genus)
                loss_family = criterion_family(p_family, t_family)
                loss = loss_species + 0.2 * loss_genus + 0.1 * loss_family
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        scheduler.step()
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} done.")

    # Prediction
    def predict(loader):
        model.eval()
        preds_list = []
        with torch.no_grad():
            for batch in loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(rank)
                with torch.cuda.amp.autocast():
                    p_species, _, _ = model(images)
                    preds = torch.argmax(p_species, dim=1)
                preds_list.append(preds.cpu())
        return torch.cat(preds_list) if preds_list else torch.tensor([])

    return_dict[f'val_{rank}'] = predict(val_loader)
    return_dict[f'test_{rank}'] = predict(test_loader)
    cleanup()

# ===== Training Functions =====

def train_convnext_multitask(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a Multi-task ConvNeXt model using DDP on 2 GPUs.
    Outputs predictions for validation and test sets.
    """
    print("Starting Multi-task ConvNeXt training engine...")
    world_size = 2
    manager = mp.Manager()
    return_dict = manager.dict()
    
    mp.spawn(
        train_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, return_dict),
        nprocs=world_size,
        join=True
    )
    
    def reconstruct(prefix, length):
        # DistributedSampler indices: [rank, rank + world_size, rank + 2*world_size, ...]
        combined_len = sum(len(return_dict[f'{prefix}_{r}']) for r in range(world_size))
        reconstructed = torch.zeros(combined_len, dtype=torch.long)
        for r in range(world_size):
            reconstructed[r::world_size] = return_dict[f'{prefix}_{r}']
        return reconstructed[:length].numpy()

    val_preds = reconstruct('val', len(X_val))
    test_preds = reconstruct('test', len(X_test))
    
    print(f"Training finished. Val size: {len(val_preds)}, Test size: {len(test_preds)}")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "convnext_multitask": train_convnext_multitask,
}
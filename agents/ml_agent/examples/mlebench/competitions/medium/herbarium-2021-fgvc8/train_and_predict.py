import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import timm
from typing import Tuple, Any, Dict, Callable

# Constants
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/herbarium-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/ec5e2488-9859-4456-b67a-20c4a0b3bb67/1/executor/output"

# Types
X = pd.DataFrame
y = pd.DataFrame
Predictions = np.ndarray
ModelFn = Callable[[X, y, X, y, X], Tuple[Predictions, Predictions]]

class HerbariumDataset(Dataset):
    def __init__(self, X_df, y_df=None, transform=None):
        self.paths = X_df['path'].values
        self.y = y_df.values if y_df is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            # Fallback for corrupted images if any
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        if self.y is not None:
            # target order: category_id, family_id, order_id
            return image, self.y[idx, 0], self.y[idx, 1], self.y[idx, 2]
        return image

class MTLConvNeXt(nn.Module):
    def __init__(self, num_classes_species, num_classes_family, num_classes_order):
        super().__init__()
        # Using ConvNeXt-Large as specified
        self.backbone = timm.create_model(
            'convnext_large.fb_in22k_ft_in1k', 
            pretrained=True, 
            num_classes=0
        )
        # Use features_only=False and num_classes=0 to get global pooled features
        feature_dim = self.backbone.num_features
        
        self.species_head = nn.Linear(feature_dim, num_classes_species)
        self.family_head = nn.Linear(feature_dim, num_classes_family)
        self.order_head = nn.Linear(feature_dim, num_classes_order)

    def forward(self, x):
        features = self.backbone(x)
        logits_species = self.species_head(features)
        logits_family = self.family_head(features)
        logits_order = self.order_head(features)
        return logits_species, logits_family, logits_order

def setup_dist(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_dist():
    dist.destroy_process_group()

def get_weighted_sampler(y_train, rank, world_size):
    # Calculate weights for species (category_id is at index 0)
    species_counts = y_train['category_id'].value_counts().to_dict()
    weights = 1.0 / np.array([species_counts[cat] for cat in y_train['category_id']])
    
    # We need a distributed version of WeightedRandomSampler.
    # Since torch doesn't provide one out-of-the-box, we pre-calculate indices for each rank.
    num_samples = len(y_train)
    indices = torch.multinomial(torch.from_numpy(weights), num_samples, replacement=True).tolist()
    
    # Split indices among ranks
    per_rank = num_samples // world_size
    rank_indices = indices[rank * per_rank : (rank + 1) * per_rank]
    
    return torch.utils.data.SubsetRandomSampler(rank_indices)

def main_worker(rank, world_size, X_train, y_train, X_val, y_val, X_test, num_classes):
    setup_dist(rank, world_size)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_ds = HerbariumDataset(X_train, y_train, train_transform)
    val_ds = HerbariumDataset(X_val, y_val, val_transform)
    test_ds = HerbariumDataset(X_test, None, val_transform)

    # Samplers
    # Using WeightedRandomSampler logic distributed across ranks
    sampler = get_weighted_sampler(y_train, rank, world_size)
    
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = MTLConvNeXt(num_classes['species'], num_classes['family'], num_classes['order']).to(rank)
    model = DDP(model, device_ids=[rank])

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # Simple linear warmup + Cosine Annealing
    num_epochs = 2 # Keeping it small for the pipeline context, but robust
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    scaler = GradScaler()

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        for images, species, family, order in train_loader:
            images = images.to(rank, non_blocking=True)
            species = species.to(rank, non_blocking=True).long()
            family = family.to(rank, non_blocking=True).long()
            order = order.to(rank, non_blocking=True).long()
            
            optimizer.zero_grad()
            with autocast():
                out_s, out_f, out_o = model(images)
                loss_s = criterion(out_s, species)
                loss_f = criterion(out_f, family)
                loss_o = criterion(out_o, order)
                loss = 1.0 * loss_s + 0.5 * loss_f + 0.2 * loss_o
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    # Prediction
    model.eval()
    def get_preds(loader):
        all_logits = []
        with torch.no_grad():
            for images in loader:
                if isinstance(images, (list, tuple)): # val loader returns labels too
                    images = images[0]
                images = images.to(rank, non_blocking=True)
                with autocast():
                    logits, _, _ = model(images)
                all_logits.append(logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    val_preds = get_preds(val_loader)
    test_preds = get_preds(test_loader)

    # Save shard to disk
    np.save(os.path.join(OUTPUT_DATA_PATH, f"val_preds_rank_{rank}.npy"), val_preds)
    np.save(os.path.join(OUTPUT_DATA_PATH, f"test_preds_rank_{rank}.npy"), test_preds)

    cleanup_dist()

def train_mtl_convnext(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a Multi-Task ConvNeXt-Large model using DDP and MTL.
    """
    print("Starting MTL ConvNeXt training with DDP...")
    
    num_classes = {
        'species': int(max(y_train['category_id'].max(), y_val['category_id'].max()) + 1),
        'family': int(max(y_train['family_id'].max(), y_val['family_id'].max()) + 1),
        'order': int(max(y_train['order_id'].max(), y_val['order_id'].max()) + 1)
    }
    
    world_size = 2
    mp.spawn(
        main_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, num_classes),
        nprocs=world_size,
        join=True
    )

    # Aggregate predictions (only rank 0's predictions are needed for val/test if we used a standard loader,
    # but since each rank saw different val/test data in a distributed setting, we'd need to gather.
    # However, in this implementation, every rank predicts on the FULL val/test set for simplicity 
    # to avoid complex gathering of indices. We'll just take rank 0's results.)
    
    val_preds = np.load(os.path.join(OUTPUT_DATA_PATH, "val_preds_rank_0.npy"))
    test_preds = np.load(os.path.join(OUTPUT_DATA_PATH, "test_preds_rank_0.npy"))

    # Cleanup temp files
    for r in range(world_size):
        os.remove(os.path.join(OUTPUT_DATA_PATH, f"val_preds_rank_{r}.npy"))
        os.remove(os.path.join(OUTPUT_DATA_PATH, f"test_preds_rank_{r}.npy"))

    print(f"Training complete. Val shape: {val_preds.shape}, Test shape: {test_preds.shape}")
    
    # Memory management
    gc.collect()
    torch.cuda.empty_cache()
    
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "mtl_convnext_large": train_mtl_convnext,
}
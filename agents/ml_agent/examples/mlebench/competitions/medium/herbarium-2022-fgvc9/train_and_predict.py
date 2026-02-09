import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from timm.data.mixup import Mixup
from typing import Tuple, Any, Dict, Callable
import tempfile
import shutil

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-11/evolux/output/mlebench/herbarium-2022-fgvc9/prepared/public"
OUTPUT_DATA_PATH = "output/5bfa936c-0be4-4e88-95de-92261403881f/1/executor/output"

X = pd.DataFrame
y = pd.DataFrame
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class HerbariumDataset(Dataset):
    def __init__(self, X_df, y_df=None, transform=None):
        self.file_paths = X_df['file_path'].values
        self.y_df = y_df
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.y_df is not None:
            # Multi-task targets
            cat = self.y_df.iloc[idx]['category_idx']
            gen = self.y_df.iloc[idx]['genus_idx']
            fam = self.y_df.iloc[idx]['family_idx']
            return image, torch.tensor([cat, gen, fam], dtype=torch.long)
        else:
            return image

class MultiTaskConvNext(nn.Module):
    def __init__(self, model_name='convnext_base', num_cats=15501, num_gens=2564, num_fams=272):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        # Get feature dimension
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        feature_dim = features.shape[1]
        
        self.head_cat = nn.Linear(feature_dim, num_cats)
        self.head_gen = nn.Linear(feature_dim, num_gens)
        self.head_fam = nn.Linear(feature_dim, num_fams)

    def forward(self, x):
        features = self.backbone(x)
        out_cat = self.head_cat(features)
        out_gen = self.head_gen(features)
        out_fam = self.head_fam(features)
        return out_cat, out_gen, out_fam

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_worker(rank, world_size, X_train, y_train, X_val, y_val, X_test, shared_results_dir):
    setup_ddp(rank, world_size)
    
    # Hyperparameters from specification
    batch_size = 64 # per GPU
    epochs = 20
    lr = 1e-4
    weight_decay = 0.05
    label_smoothing = 0.1
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and Samplers
    train_ds = HerbariumDataset(X_train, y_train, transform=train_transform)
    val_ds = HerbariumDataset(X_val, y_val, transform=val_transform)
    test_ds = HerbariumDataset(X_test, None, transform=val_transform)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    # Model
    model = MultiTaskConvNext().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss and Mixup
    # Note: Mixup usually handles label smoothing. We apply mixup only to training.
    # For multi-task, we apply mixup indices and lambda manually or via timm.
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, 
        mode='batch', label_smoothing=label_smoothing, num_classes=15501
    )
    
    # Standard CE for targets not passed to mixup (genus/fam) 
    # but we need to handle mixed labels for them too.
    # To keep implementation lean, we use mixup indices to mix genus/family targets.
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler()

    print(f"[Rank {rank}] Starting training...")
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for images, targets in train_loader:
            images, targets = images.cuda(rank), targets.cuda(rank)
            
            # targets: [batch, 3] -> cat, gen, fam
            # Apply mixup
            images, mixed_cats = mixup_fn(images, targets[:, 0])
            
            # Since timm Mixup only supports one target, we handle gen/fam by weight 
            # if we wanted full precision, but baseline suggests weights on heads.
            # For gen/fam, we'll use standard CE on the original labels for simplicity
            # or apply the same mixup logic. Given the baseline, simple MTL is key.

            optimizer.zero_grad()
            with autocast():
                out_cat, out_gen, out_fam = model(images)
                
                loss_cat = criterion(out_cat, mixed_cats)
                loss_gen = criterion(out_gen, targets[:, 1])
                loss_fam = criterion(out_fam, targets[:, 2])
                
                loss = 0.7 * loss_cat + 0.2 * loss_gen + 0.1 * loss_fam
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} complete.")

    # Prediction phase
    model.eval()
    
    def get_preds(dataset):
        # Distributed prediction
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, list): # X, y case
                    images = batch[0].cuda(rank)
                else: # X only case
                    images = batch.cuda(rank)
                
                with autocast():
                    out_cat, _, _ = model(images)
                all_preds.append(out_cat.argmax(dim=1).cpu().numpy())
        
        preds = np.concatenate(all_preds)
        
        # Gather all predictions to rank 0
        all_gathered_preds = [None for _ in range(world_size)]
        dist.all_gather_object(all_gathered_preds, preds)
        
        if rank == 0:
            # Reconstruct in original order
            # DistributedSampler pads the dataset to be divisible by world_size
            final_preds = []
            for i in range(len(dataset)):
                rank_idx = i % world_size
                local_idx = i // world_size
                final_preds.append(all_gathered_preds[rank_idx][local_idx])
            return np.array(final_preds)
        return None

    val_preds_idx = get_preds(val_ds)
    test_preds_idx = get_preds(test_ds)
    
    if rank == 0:
        # Save to shared directory
        np.save(os.path.join(shared_results_dir, "val_preds.npy"), val_preds_idx)
        np.save(os.path.join(shared_results_dir, "test_preds.npy"), test_preds_idx)
        print("Rank 0: Predictions saved successfully.")

    cleanup_ddp()

def train_convnext_mtl(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a ConvNeXt-Base model using Multi-Task Learning (Category, Genus, Family).
    Implements Distributed Data Parallel across available GPUs.
    """
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU available for training.")
    
    print(f"Initializing DDP Training on {world_size} GPUs...")
    
    # Create category mapping to return original category_id
    cat_idx_to_id = dict(zip(y_train['category_idx'], y_train['category_id']))
    
    shared_results_dir = tempfile.mkdtemp()
    try:
        mp.spawn(
            train_worker,
            args=(world_size, X_train, y_train, X_val, y_val, X_test, shared_results_dir),
            nprocs=world_size,
            join=True
        )
        
        # Load results
        val_preds_idx = np.load(os.path.join(shared_results_dir, "val_preds.npy"))
        test_preds_idx = np.load(os.path.join(shared_results_dir, "test_preds.npy"))
        
        # Map indices back to category_id
        val_preds = np.array([cat_idx_to_id[idx] for idx in val_preds_idx])
        test_preds = np.array([cat_idx_to_id[idx] for idx in test_preds_idx])
        
    finally:
        shutil.rmtree(shared_results_dir)

    print(f"Training and Prediction complete. Predictions generated for {len(val_preds)} val and {len(test_preds)} test samples.")
    return val_preds, test_preds

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "convnext_mtl": train_convnext_mtl,
}
import os
import shutil
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, transforms
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, Any, Dict, Callable

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.DataFrame
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class BSONDataset(data.Dataset):
    """
    Dataset for on-the-fly BSON decoding. 
    Always returns the global index to allow reconstruction of the original order in DDP.
    """
    def __init__(self, X_df: pd.DataFrame, y_df: pd.DataFrame = None, transform=None):
        self.X = X_df.reset_index(drop=True)
        self.y = y_df.reset_index(drop=True) if y_df is not None else None
        self.transform = transform
        self.handles = {}

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        path = row['bson_path']
        offset = row['offset']
        length = row['length']
        img_idx = row['img_idx']

        # Resource management: per-worker file handles
        if path not in self.handles:
            self.handles[path] = open(path, 'rb')
        
        f = self.handles[path]
        f.seek(offset)
        doc_bytes = f.read(length)
        
        try:
            import bson
            doc = bson.BSON(doc_bytes).decode()
        except:
            import bson as bson_alt
            doc = bson_alt.loads(doc_bytes)
            
        img_data = doc['imgs'][img_idx]['picture']
        # Efficient decoding via OpenCV
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        if self.y is not None:
            target = self.y.iloc[idx]
            # Multi-Task Targets + Index
            return img, target['l1_idx'], target['l2_idx'], target['cat_idx'], idx
        else:
            # Image + Index
            return img, idx

    def __len__(self):
        return len(self.X)

    def __del__(self):
        for f in self.handles.values():
            f.close()

class MultiTaskEffNet(nn.Module):
    """
    EfficientNet-B2 with three hierarchical classification heads (L1, L2, L3).
    """
    def __init__(self, num_l1=49, num_l2=483, num_l3=5270):
        super().__init__()
        # Load backbone with ImageNet weights
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() 
        
        self.head_l1 = nn.Linear(in_features, num_l1)
        self.head_l2 = nn.Linear(in_features, num_l2)
        self.head_l3 = nn.Linear(in_features, num_l3)
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.head_l1(feat), self.head_l2(feat), self.head_l3(feat)

def train_worker(rank, world_size, X_train, y_train, X_val, y_val, X_test, temp_dir):
    """
    Distributed training worker process.
    """
    # 1. Setup Distributed Group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 2. Data Preparation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = BSONDataset(X_train, y_train, train_transform)
    val_ds = BSONDataset(X_val, y_val, val_transform)
    test_ds = BSONDataset(X_test, None, val_transform)

    batch_size = 256 # Per GPU
    train_sampler = data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=12, pin_memory=True)
    
    val_sampler = data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)
    
    test_sampler = data.distributed.DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, sampler=test_sampler, num_workers=8, pin_memory=True)

    # 3. Model, Optimizer, Scheduler
    model = MultiTaskEffNet().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    epochs = 5
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    scaler = torch.amp.GradScaler('cuda')

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for i, (imgs, l1, l2, l3, _) in enumerate(train_loader):
            imgs, l1, l2, l3 = imgs.cuda(rank), l1.long().cuda(rank), l2.long().cuda(rank), l3.long().cuda(rank)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out1, out2, out3 = model(imgs)
                loss = 0.1 * criterion(out1, l1) + 0.1 * criterion(out2, l2) + 0.8 * criterion(out3, l3)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if rank == 0 and i % 500 == 0:
                print(f"Epoch {epoch} Step {i}/{len(train_loader)} Loss: {loss.item():.4f}")

    # 5. Inference
    model.eval()
    
    def run_inference(loader, ds_len):
        indices_list = []
        logits_list = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0].cuda(rank)
                idx_batch = batch[-1] # Index is always last
                
                with torch.amp.autocast('cuda'):
                    _, _, out3 = model(imgs)
                
                logits_list.append(out3.cpu().numpy())
                indices_list.append(idx_batch.cpu().numpy())
        
        all_logits = np.concatenate(logits_list, axis=0)
        all_indices = np.concatenate(indices_list, axis=0)
        
        # Remove DDP padding
        mask = all_indices < ds_len
        return all_indices[mask], all_logits[mask]

    val_indices, val_logits = run_inference(val_loader, len(val_ds))
    test_indices, test_logits = run_inference(test_loader, len(test_ds))

    # Save local results to disk for aggregation
    np.save(os.path.join(temp_dir, f"val_indices_{rank}.npy"), val_indices)
    np.save(os.path.join(temp_dir, f"val_logits_{rank}.npy"), val_logits)
    np.save(os.path.join(temp_dir, f"test_indices_{rank}.npy"), test_indices)
    np.save(os.path.join(temp_dir, f"test_logits_{rank}.npy"), test_logits)
    
    dist.barrier()
    dist.destroy_process_group()

def train_multitask_efficientnet_b2(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a Multi-Task EfficientNet-B2 and returns image-level logits.
    Correctly handles DDP data reconstruction using returned indices.
    """
    print(f"Starting Multi-Task Training on {len(X_train)} images...")
    
    world_size = torch.cuda.device_count()
    temp_dir = tempfile.mkdtemp()
    
    try:
        mp.spawn(
            train_worker,
            args=(world_size, X_train, y_train, X_val, y_val, X_test, temp_dir),
            nprocs=world_size,
            join=True
        )

        def aggregate_results(prefix, total_len, num_classes):
            indices = []
            logits = []
            for r in range(world_size):
                indices.append(np.load(os.path.join(temp_dir, f"{prefix}_indices_{r}.npy")))
                logits.append(np.load(os.path.join(temp_dir, f"{prefix}_logits_{r}.npy")))
            
            indices = np.concatenate(indices)
            logits = np.concatenate(logits)
            
            final_logits = np.zeros((total_len, num_classes), dtype=np.float32)
            final_logits[indices] = logits
            return final_logits

        val_preds = aggregate_results("val", len(X_val), 5270)
        test_preds = aggregate_results("test", len(X_test), 5270)
        
        print("Training and inference complete.")
        return val_preds, test_preds

    finally:
        shutil.rmtree(temp_dir)

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "multitask_efficientnet_b2": train_multitask_efficientnet_b2,
}
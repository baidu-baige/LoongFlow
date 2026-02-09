import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import timm
import pandas as pd
import numpy as np
import cv2
import pickle
import os
import math
from typing import Tuple, Any, Dict, Callable
from sklearn.preprocessing import LabelEncoder

# Path Configuration
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/hotel-id-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/6c275358-248b-46e3-a3f8-feb17fef7b7f/3/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Model Components =====

class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) - SOTA for retrieval tasks.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) to maximize inter-class separation.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class HotelModel(nn.Module):
    """
    Embedding model with EfficientNet-V2-S backbone, GeM, ArcFace, and Auxiliary Chain head.
    """
    def __init__(self, n_classes, n_chains, model_name='tf_efficientnetv2_s_in21k', embedding_size=512):
        super(HotelModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        in_features = self.backbone.num_features
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.arcface = ArcMarginProduct(embedding_size, n_classes, s=30, m=0.5)
        self.chain_head = nn.Linear(embedding_size, n_chains)

    def forward(self, x, label=None):
        features = self.backbone(x)
        features = self.pooling(features).view(features.size(0), -1)
        embedding = self.bn(self.embedding(features))
        
        logits_hotel = self.arcface(embedding, label)
        logits_chain = self.chain_head(embedding)
        
        return logits_hotel, logits_chain, F.normalize(embedding)

# ===== Dataset =====

class HotelDataset(Dataset):
    def __init__(self, df, labels=None, chain_labels=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = labels.values if hasattr(labels, 'values') else labels
        self.chain_labels = chain_labels.values if hasattr(chain_labels, 'values') else chain_labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'image_path']
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((384, 384, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if self.labels is not None:
            return image, torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(self.chain_labels[idx], dtype=torch.long)
        return image

# ===== Training Orchestration =====

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, train_data, val_data, test_data, results_dict):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    X_train_fold, y_train_fold, y_train_chain_fold, num_hotels, num_chains, X_full_train = train_data
    X_val, X_test = val_data, test_data

    # Load transformations from Stage 3
    with open(os.path.join(OUTPUT_DATA_PATH, "train_transform.pkl"), "rb") as f:
        train_transform = pickle.load(f)
    with open(os.path.join(OUTPUT_DATA_PATH, "val_transform.pkl"), "rb") as f:
        val_transform = pickle.load(f)

    # Data Loading
    train_ds = HotelDataset(X_train_fold, y_train_fold, y_train_chain_fold, transform=train_transform)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=train_sampler, num_workers=8, pin_memory=True)

    model = HotelModel(num_hotels, num_chains).cuda(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion_chain = nn.CrossEntropyLoss()
    criterion_hotel = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training: 10 Epochs
    for epoch in range(10):
        model.train()
        train_sampler.set_epoch(epoch)
        for images, hotel_labels, chain_labels in train_loader:
            images, hotel_labels, chain_labels = images.cuda(rank), hotel_labels.cuda(rank), chain_labels.cuda(rank)
            optimizer.zero_grad()
            with autocast():
                logits_hotel, logits_chain, _ = model(images, hotel_labels)
                loss = 0.8 * criterion_hotel(logits_hotel, hotel_labels) + 0.2 * criterion_chain(logits_chain, chain_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if rank == 0:
            print(f"Epoch {epoch+1}/10 complete.")

    # Inference and Gallery Extraction
    model.eval()
    
    def get_distributed_outputs(df):
        ds = HotelDataset(df, transform=val_transform)
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(ds, batch_size=128, sampler=sampler, num_workers=8, pin_memory=True)
        
        embeddings, logits_list = [], []
        with torch.no_grad():
            for images in loader:
                images = images.cuda(rank)
                with autocast():
                    logits, _, embs = model(images)
                embeddings.append(embs.cpu().numpy())
                logits_list.append(logits.cpu().numpy())
        
        local_embs = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 512))
        local_logits = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, num_hotels))
        
        all_embs = [None for _ in range(world_size)]
        all_logits = [None for _ in range(world_size)]
        dist.all_gather_object(all_embs, local_embs)
        dist.all_gather_object(all_logits, local_logits)
        
        final_embs = np.concatenate(all_embs, axis=0)[:len(df)]
        final_logits = np.concatenate(all_logits, axis=0)[:len(df)]
        return final_embs, final_logits

    # Full Training Gallery Extraction
    gallery_embs, _ = get_distributed_outputs(X_full_train)
    
    # Validation and Test Predictions
    _, val_logits = get_distributed_outputs(X_val)
    _, test_logits = get_distributed_outputs(X_test)

    if rank == 0:
        np.save(os.path.join(OUTPUT_DATA_PATH, "train_gallery_embeddings.npy"), gallery_embs)
        results_dict['val_preds'] = np.argsort(val_logits, axis=1)[:, -5:][:, ::-1]
        results_dict['test_preds'] = np.argsort(test_logits, axis=1)[:, -5:][:, ::-1]
        torch.save(model.module.state_dict(), os.path.join(OUTPUT_DATA_PATH, "hotel_model.pth"))

    dist.barrier()
    cleanup()

def train_retrieval_arcface_efficientnet(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    print("Stage 4: Training EfficientNet-V2 ArcFace Retrieval Model...")
    
    def extract_chain(path):
        parts = path.split(os.sep)
        return parts[-2] if len(parts) > 1 else "0"

    X_full_train = pd.concat([X_train, X_val]).reset_index(drop=True)
    chain_le = LabelEncoder()
    chain_le.fit(X_full_train['image_path'].apply(extract_chain))
    
    y_train_chain_fold = chain_le.transform(X_train['image_path'].apply(extract_chain))
    num_hotels = int(y_train.max() + 1)
    num_chains = len(chain_le.classes_)
    world_size = torch.cuda.device_count()
    
    manager = mp.Manager()
    results_dict = manager.dict()
    train_data = (X_train, y_train, y_train_chain_fold, num_hotels, num_chains, X_full_train)
    
    mp.spawn(train_worker, args=(world_size, train_data, X_val, X_test, results_dict), nprocs=world_size, join=True)
    
    return results_dict['val_preds'], results_dict['test_preds']

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "retrieval_arcface_efficientnet": train_retrieval_arcface_efficientnet,
}
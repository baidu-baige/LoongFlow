import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os
import tempfile
from typing import List, Dict, Any, Tuple, Callable

# Task-adaptive type definitions
X = List[np.ndarray]  # List of T x 555 feature matrices (363 motion + 192 audio)
y = List[np.ndarray]  # List of T-length label sequences (0-20)
Predictions = List[np.ndarray]  # List of T x 21 probability matrices

ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-02/evolux/output/mlebench/multi-modal-gesture-recognition/prepared/public"
OUTPUT_DATA_PATH = "output/7d9b4fa5-39b3-4a58-b088-17228f41073d/11/executor/output"

# ===== Model Components =====

class MSTC(nn.Module):
    """Multi-Scale Temporal CNN branch with kernel sizes [3, 5, 7, 9]."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        branch_dim = out_dim // 4
        self.conv3 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_dim, branch_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_dim, branch_dim, kernel_size=7, padding=3)
        self.conv9 = nn.Conv1d(in_dim, branch_dim, kernel_size=9, padding=4)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        x = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x), self.conv9(x)], dim=1)
        return self.relu(self.bn(x))

class ModalityGating(nn.Module):
    """Gated Linear Units to adaptively weight motion vs. audio features."""
    def __init__(self, motion_dim: int, audio_dim: int, hid_dim: int):
        super().__init__()
        # Map each modality to same hidden dimension for fusion
        self.m_gate = nn.Linear(motion_dim, hid_dim * 2)
        self.a_gate = nn.Linear(audio_dim, hid_dim * 2)
        
    def forward(self, motion: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        # GLU: x * sigmoid(gate)
        m_f = F.glu(self.m_gate(motion), dim=-1)
        a_f = F.glu(self.a_gate(audio), dim=-1)
        return torch.cat([m_f, a_f], dim=-1) # (B, T, hid_dim * 2)

class MSTC_BiGRU_Attn(nn.Module):
    def __init__(self, hid_dim: int = 256, num_classes: int = 21):
        super().__init__()
        # 1. Modality Gating (363 motion, 192 audio)
        self.gating = ModalityGating(363, 192, hid_dim // 2)
        
        # 2. Multi-Scale Temporal CNN
        self.mstc = MSTC(hid_dim, hid_dim)
        
        # 3. Temporal Module: Bi-GRU
        self.gru = nn.GRU(hid_dim, hid_dim // 2, num_layers=2, 
                          bidirectional=True, batch_first=True, dropout=0.2)
        
        # 4. Attention (4 heads)
        self.attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=4, batch_first=True)
        
        # 5. Dual Heads (Class, Boundary)
        self.classifier = nn.Linear(hid_dim, num_classes)
        self.boundary_head = nn.Linear(hid_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 555)
        motion = x[:, :, :363]
        audio = x[:, :, 363:]
        
        # Modality Gating
        fused = self.gating(motion, audio) # (B, T, hid_dim)
        
        # Spatial-to-Temporal
        fused = fused.transpose(1, 2) # (B, hid_dim, T)
        fused = self.mstc(fused)
        fused = fused.transpose(1, 2) # (B, T, hid_dim)
        
        # Temporal Bi-GRU
        gru_out, _ = self.gru(fused)
        
        # Multi-Head Attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        out = gru_out + attn_out
        
        logits_cls = self.classifier(out)
        logits_boundary = self.boundary_head(out).squeeze(-1)
        return logits_cls, logits_boundary

# ===== Data Handling & Augmentation =====

class GestureDataset(Dataset):
    def __init__(self, X_data: X, y_data: y = None, augment: bool = False):
        self.X = X_data
        self.y = y_data
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.X[idx].copy()
        
        if self.augment:
            # Gaussian noise and scaling on motion features (first 363 dims)
            noise = np.random.normal(0, 0.01, (feat.shape[0], 363)).astype(np.float32)
            feat[:, :363] += noise
            scale = np.random.uniform(0.9, 1.1)
            feat[:, :363] *= scale
            
        feat_tensor = torch.from_numpy(feat).float()
        
        if self.y is not None:
            labels = self.y[idx]
            T = len(labels)
            
            # Boundary-weighted mask for Cross Entropy
            b_mask = np.ones(T, dtype=np.float32)
            # Boundary targets for binary classification head
            b_target = np.zeros(T, dtype=np.float32)
            
            changes = np.where(labels[:-1] != labels[1:])[0]
            for cp in changes:
                # CE Weighting: 2.0 for frames near transitions
                b_mask[max(0, cp-2):min(T, cp+3)] = 2.0
                # Boundary Binary Target: 1.0 for frames at transition
                b_target[max(0, cp-1):min(T, cp+2)] = 1.0

            return feat_tensor, torch.from_numpy(labels).long(), torch.from_numpy(b_mask).float(), torch.from_numpy(b_target).float()
        
        return feat_tensor, torch.zeros(feat.shape[0], dtype=torch.long), torch.ones(feat.shape[0]), torch.zeros(feat.shape[0])

def collate_fn(batch):
    feats, labels, b_masks, b_targets = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats])
    max_len = lengths.max().item()
    
    padded_feats = torch.zeros(len(feats), max_len, feats[0].shape[1])
    padded_labels = torch.full((len(labels), max_len), -100) # ignore_index
    padded_b_masks = torch.ones(len(b_masks), max_len)
    padded_b_targets = torch.zeros(len(b_targets), max_len)
    
    for i, (f, l, m, t) in enumerate(zip(feats, labels, b_masks, b_targets)):
        padded_feats[i, :lengths[i], :] = f
        padded_labels[i, :lengths[i]] = l
        padded_b_masks[i, :lengths[i]] = m
        padded_b_targets[i, :lengths[i]] = t
        
    return padded_feats, padded_labels, padded_b_masks, padded_b_targets

# ===== Loss Logic =====

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: torch.Tensor = None, weights: torch.Tensor = None):
    """Focal Loss with per-frame boundary weighting."""
    mask = (targets != -100)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
    # Standard CE with per-class alpha
    ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=alpha, ignore_index=-100)
    
    # Calculate probability for focal term
    pt = torch.exp(-ce_loss)
    f_loss = (1 - pt) ** gamma * ce_loss
    
    # Apply per-frame boundary weights (transition regions)
    if weights is not None:
        f_loss = f_loss * weights
        
    # Return mean only over valid frames
    return f_loss[mask].mean()

# ===== Training Worker =====

def train_worker(rank: int, world_size: int, X_train: X, y_train: y, alpha: torch.Tensor, model_path: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12399'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    train_ds = GestureDataset(X_train, y_train, augment=True)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=16, sampler=train_sampler, collate_fn=collate_fn, num_workers=8, pin_memory=True)
    
    model = MSTC_BiGRU_Attn().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()
    criterion_b = nn.BCEWithLogitsLoss()
    
    epochs = 40
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        for feats, labels, b_masks, b_targets in train_loader:
            feats, labels, b_masks, b_targets = feats.to(device), labels.to(device), b_masks.to(device), b_targets.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits_cls, logits_b = model(feats)
                
                # Classification Path: Focal Loss + Boundary Weights
                loss_cls = focal_loss(
                    logits_cls.view(-1, 21), 
                    labels.view(-1), 
                    gamma=2.0, 
                    alpha=alpha.to(device), 
                    weights=b_masks.view(-1)
                )
                
                # Boundary Detection Path: BCE Loss
                valid_mask = (labels != -100)
                loss_boundary = criterion_b(logits_b[valid_mask], b_targets[valid_mask])
                
                total_loss = loss_cls + 0.5 * loss_boundary
                
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()
            
        scheduler.step()
        if rank == 0 and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Rank 0 | Loss: {epoch_loss/len(train_loader):.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), model_path)
    
    dist.barrier()
    dist.destroy_process_group()

# ===== Training Orchestrator =====

def train_mstc_gru_fusion(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Implementation of Multi-Scale Temporal CNN + Bi-GRU + Attention with adaptive fusion.
    """
    print(f"Initializing MSTC-GRU-Fusion Ensemble on {torch.cuda.device_count()} H20 GPUs...")
    
    # Precompute per-class alpha based on inverse square root frequency
    all_y = np.concatenate(y_train)
    counts = np.bincount(all_y, minlength=21)
    alpha = 1.0 / (np.sqrt(counts) + 1.0)
    alpha_tensor = torch.from_numpy(alpha / alpha.sum() * 21).float()
    
    fd, model_path = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    
    world_size = 2
    try:
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, X_train, y_train, alpha_tensor, model_path))
    except Exception as e:
        if os.path.exists(model_path): os.remove(model_path)
        raise e
        
    print("Optimization complete. Running sequential inference...")
    device = torch.device("cuda:0")
    model = MSTC_BiGRU_Attn().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    def inference(data: X) -> Predictions:
        preds = []
        with torch.no_grad():
            for sample in data:
                feat = torch.from_numpy(sample).float().unsqueeze(0).to(device)
                with torch.cuda.amp.autocast():
                    logits_cls, _ = model(feat)
                    probs = torch.softmax(logits_cls, dim=-1).squeeze(0)
                preds.append(probs.cpu().numpy())
        return preds

    val_preds = inference(X_val)
    test_preds = inference(X_test)
    
    if os.path.exists(model_path):
        os.remove(model_path)
        
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "mstc_gru_fusion": train_mstc_gru_fusion,
}
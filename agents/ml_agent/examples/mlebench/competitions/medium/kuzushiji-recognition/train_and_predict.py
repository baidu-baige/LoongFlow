import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import cv2
import timm
from typing import Tuple, Any, Dict, List, Callable
from tqdm import tqdm

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/kuzushiji-recognition/prepared/public"
OUTPUT_DATA_PATH = "output/2cf35106-db22-4d8d-a450-9ac60aada454/1/executor/output"

# Task-adaptive type definitions
X = List[Dict[str, Any]]
y = List[Dict[str, Any]]
Predictions = List[str]
ModelFn = Callable[[X, y, X, y, X], Tuple[Predictions, Predictions]]

# ===== Models =====

class CenterNet(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(CenterNet, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        # ResNet50 strides: 4, 8, 16, 32. Channels: 256, 512, 1024, 2048
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Heatmap (1 channel for "character center")
        self.heatmap = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        # Offset (2 channels: dx, dy)
        self.offset = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        up = self.upsample(features[-1])
        hm = self.heatmap(up)
        offset = self.offset(up)
        return hm, offset

class Classifier(nn.Module):
    def __init__(self, num_classes=4113):
        super(Classifier, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# ===== Datasets =====

class DetectionDataset(Dataset):
    def __init__(self, X_data, y_data=None, is_train=True):
        self.tiles = []
        self.labels = []
        for img_idx, img_dict in enumerate(X_data):
            for tile_idx, tile_img in enumerate(img_dict['tiles']):
                self.tiles.append(tile_img)
                if y_data:
                    self.labels.append(y_data[img_idx]['tile_labels'][tile_idx])
                else:
                    self.labels.append([])
        self.is_train = is_train

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        tile = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
        
        if not self.is_train:
            return tile

        hm = np.zeros((256, 256), dtype=np.float32)
        offset = np.zeros((2, 256, 256), dtype=np.float32)
        
        for lbl in self.labels[idx]:
            # Centernet heatmap logic
            ctx, cty = lbl['rx'] / 4.0, lbl['ry'] / 4.0
            ix, iy = int(ctx), int(cty)
            if 0 <= ix < 256 and 0 <= iy < 256:
                self.draw_gaussian(hm, (ix, iy), 2.0)
                offset[0, iy, ix] = ctx - ix
                offset[1, iy, ix] = cty - iy
                
        return tile, torch.from_numpy(hm).unsqueeze(0), torch.from_numpy(offset)

    def draw_gaussian(self, hm, center, sigma):
        tmp_size = sigma * 3
        mu_x, mu_y = center
        w, h = hm.shape[1], hm.shape[0]
        x_min, y_min = int(max(0, mu_x - tmp_size)), int(max(0, mu_y - tmp_size))
        x_max, y_max = int(min(w, mu_x + tmp_size + 1)), int(min(h, mu_y + tmp_size + 1))
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                v = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))
                hm[y, x] = max(hm[y, x], v)

class ClassificationDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.crops = []
        self.labels = []
        for img_idx, img_dict in enumerate(X_data):
            for crop_idx, crop_img in enumerate(img_dict['crops']):
                self.crops.append(crop_img)
                self.labels.append(y_data[img_idx]['crop_labels'][crop_idx])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        label = self.labels[idx]
        return crop, label

# ===== Losses & Helpers =====

def focal_loss(pred, target, alpha=2, beta=4):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    neg_weights = torch.pow(1 - target, beta)

    loss = 0
    pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===== Training Logic =====

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    torch.distributed.destroy_process_group()

def train_det(rank, world_size, train_data, det_model_path):
    setup_ddp(rank, world_size)
    
    dataset = DetectionDataset(train_data[0], train_data[1], is_train=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4)
    
    model = CenterNet().cuda()
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(50):
        model.train()
        sampler.set_epoch(epoch)
        for tiles, hms, offsets in loader:
            tiles, hms, offsets = tiles.cuda(), hms.cuda(), offsets.cuda()
            optimizer.zero_grad()
            pred_hm, pred_off = model(tiles)
            loss_hm = focal_loss(pred_hm, hms)
            
            # Masked L1 loss for offsets
            mask = hms.eq(1).float()
            loss_off = F.l1_loss(pred_off * mask, offsets * mask, reduction='sum') / (mask.sum() + 1e-6)
            
            loss = loss_hm + loss_off
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"Det Epoch {epoch+1}/50 Loss: {loss.item():.4f}")
            
    if rank == 0:
        torch.save(model.module.state_dict(), det_model_path)
    cleanup_ddp()

def train_cls(rank, world_size, train_data, cls_model_path):
    setup_ddp(rank, world_size)
    
    dataset = ClassificationDataset(train_data[0], train_data[1])
    # For DDP classification, we'll use DistributedSampler
    dist_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=dist_sampler, num_workers=4)
    
    model = Classifier(num_classes=4113).cuda()
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for epoch in range(30):
        model.train()
        dist_sampler.set_epoch(epoch)
        for crops, labels in loader:
            crops, labels = crops.cuda(), labels.cuda()
            crops, y_a, y_b, lam = mixup_data(crops, labels, alpha=0.2)
            optimizer.zero_grad()
            outputs = model(crops)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"Cls Epoch {epoch+1}/30 Loss: {loss.item():.4f}")
            
    if rank == 0:
        torch.save(model.module.state_dict(), cls_model_path)
    cleanup_ddp()

# ===== Main Function =====

def train_centernet_efficientnet(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    print("Starting Training Stage...")
    
    det_model_path = os.path.join(OUTPUT_DATA_PATH, "det_model.pth")
    cls_model_path = os.path.join(OUTPUT_DATA_PATH, "cls_model.pth")
    
    # Spawn DDP training for Detector (2 GPUs)
    mp.spawn(train_det, args=(2, (X_train, y_train), det_model_path), nprocs=2, join=True)
    
    # Spawn DDP training for Classifier (2 GPUs)
    mp.spawn(train_cls, args=(2, (X_train, y_train), cls_model_path), nprocs=2, join=True)
    
    # Load models for inference
    det_model = CenterNet().cuda().eval()
    det_model.load_state_dict(torch.load(det_model_path))
    
    cls_model = Classifier(num_classes=4113).cuda().eval()
    cls_model.load_state_dict(torch.load(cls_model_path))
    
    # Rebuild unicode_map_inv from classification metadata in BASE_DATA_PATH
    class_meta_path = os.path.join(BASE_DATA_PATH, "classification_metadata.csv")
    df_class = pd.read_csv(class_meta_path)
    unique_unicodes = sorted(df_class['unicode'].unique())
    unicode_map_inv = {i: u for i, u in enumerate(unique_unicodes)}

    def detect_and_classify(X_data):
        results = []
        with torch.no_grad():
            for img_dict in tqdm(X_data, desc="Inference"):
                img_preds = []
                # STRIDE = 1024 - 128 = 896
                STRIDE = 896
                H, W = img_dict['orig_shape']
                
                tiles_x = []
                x = 0
                while x + 1024 < W:
                    tiles_x.append(x)
                    x += STRIDE
                tiles_x.append(max(0, W - 1024))
                tiles_x = sorted(list(set(tiles_x)))
                
                tiles_y = []
                y = 0
                while y + 1024 < H:
                    tiles_y.append(y)
                    y += STRIDE
                tiles_y.append(max(0, H - 1024))
                tiles_y = sorted(list(set(tiles_y)))
                
                tile_idx = 0
                for ty in tiles_y:
                    for tx in tiles_x:
                        if tile_idx >= len(img_dict['tiles']): break
                        tile_img = img_dict['tiles'][tile_idx]
                        tile_idx += 1
                        
                        tile_tensor = torch.from_numpy(tile_img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
                        hm, off = det_model(tile_tensor)
                        
                        # Peak find
                        hm_pool = F.max_pool2d(hm, 3, stride=1, padding=1)
                        peaks = (hm == hm_pool) & (hm > 0.3)
                        peaks = peaks.squeeze().cpu().numpy()
                        ys, xs = np.where(peaks)
                        
                        if len(xs) > 0:
                            crops = []
                            coords = []
                            for py, px in zip(ys, xs):
                                gx = int(tx + (px + off[0, 0, py, px].item()) * 4)
                                gy = int(ty + (py + off[0, 1, py, px].item()) * 4)
                                
                                c_tx, c_ty = int(px * 4), int(py * 4)
                                x1, y1 = max(0, c_tx - 64), max(0, c_ty - 64)
                                x2, y2 = min(1024, c_tx + 64), min(1024, c_ty + 64)
                                crop = tile_img[y1:y2, x1:x2]
                                crop = cv2.resize(crop, (128, 128))
                                crops.append(torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0)
                                coords.append((gx, gy))
                                
                            if crops:
                                crops_tensor = torch.stack(crops).cuda()
                                cls_outputs = cls_model(crops_tensor)
                                cls_labels = cls_outputs.argmax(dim=1).cpu().numpy()
                                
                                for label_id, (gx, gy) in zip(cls_labels, coords):
                                    unicode = unicode_map_inv.get(label_id, "")
                                    if unicode:
                                        img_preds.append(f"{unicode} {gx} {gy}")
                
                results.append(" ".join(img_preds))
        return results

    val_preds = detect_and_classify(X_val)
    test_preds = detect_and_classify(X_test)
    
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "centernet_efficientnet": train_centernet_efficientnet,
}
import os
import gc
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Any, Dict, List, Callable
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/3d-object-detection-for-autonomous-vehicles/prepared/public"
OUTPUT_DATA_PATH = "output/341879f4-a7a1-4d18-a801-6d16eb36af1b/1/executor/output"

# Task-adaptive type definitions
X = List[Dict[str, Any]]
y = List[List[Dict[str, Any]]]
Predictions = List[List[Dict[str, Any]]]

ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

CLASS_MAP = {
    'car': 0, 'other_vehicle': 1, 'pedestrian': 2, 'bicycle': 3, 'truck': 4,
    'bus': 5, 'motorcycle': 6, 'emergency_vehicle': 7, 'animal': 8
}
REV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

PC_RANGE = [-100.0, -100.0, -5.0, 100.0, 100.0, 3.0]
VOXEL_SIZE = [0.2, 0.2, 8.0]
GRID_SIZE = [1000, 1000, 1] # x, y, z
DOWNSAMPLE = 4
FEATURE_MAP_SIZE = [250, 250] # y, x

# ===== Helper Functions =====

def quat_to_rot_mat(q: List[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_yaw_from_quat(q: List[float]) -> float:
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = np.zeros((diameter, diameter))
    sigma = diameter / 6
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    
    left, right = min(center[0], radius), min(heatmap.shape[1] - center[0] - 1, radius)
    top, bottom = min(center[1], radius), min(heatmap.shape[0] - center[1] - 1, radius)

    masked_heatmap = heatmap[center[1]-top:center[1]+bottom+1, center[0]-left:center[0]+right+1]
    masked_gaussian = gaussian[radius-top:radius+bottom+1, radius-left:radius+right+1]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

# ===== Model Components =====

class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, voxels, num_points):
        x = self.net(voxels.view(-1, voxels.shape[-1]))
        x = x.view(voxels.shape[0], voxels.shape[1], -1)
        x = x.max(dim=1)[0]
        return x

class PointPillarsScatter(nn.Module):
    def __init__(self, in_channels=64, grid_size=GRID_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
    def forward(self, features, coords, batch_size):
        # features: (M, C), coords: (M, 4) [batch_idx, z, y, x]
        canvas = torch.zeros(self.in_channels, batch_size * self.grid_size[1] * self.grid_size[0], 
                             device=features.device, dtype=features.dtype)
        
        indices = coords[:, 0].long() * (self.grid_size[1] * self.grid_size[0]) + \
                  coords[:, 2].long() * self.grid_size[0] + \
                  coords[:, 3].long()
        
        indices = indices.unsqueeze(0).expand(self.in_channels, -1)
        canvas.scatter_(1, indices, features.T)
        
        return canvas.view(self.in_channels, batch_size, self.grid_size[1], self.grid_size[0]).permute(1, 0, 2, 3)

class CenterPointBackbone(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        # Target feature map size is 250x250.
        # block1 output is 500x500. block2 output is 250x250.
        self.up1 = nn.Conv2d(64, 64, 3, stride=2, padding=1) # 500 -> 250
        self.up2 = nn.Conv2d(128, 64, 3, stride=1, padding=1) # 250 -> 250
        
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        return torch.cat([self.up1(x1), self.up2(x2)], dim=1)

class CenterHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=9):
        super().__init__()
        self.heatmap = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, num_classes, 1))
        self.offset = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.z = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.dim = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 3, 1))
        self.yaw = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
    def forward(self, x):
        return {
            'heatmap': torch.sigmoid(self.heatmap(x)),
            'offset': self.offset(x),
            'z': self.z(x),
            'dim': self.dim(x),
            'yaw': self.yaw(x)
        }

class CenterPointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.scatter = PointPillarsScatter()
        self.backbone = CenterPointBackbone()
        self.head = CenterHead()
    def forward(self, voxels, num_points, coords, batch_size):
        x = self.pfn(voxels, num_points)
        x = self.scatter(x, coords, batch_size)
        x = self.backbone(x)
        return self.head(x)

# ===== Dataset =====

class CenterPointDataset(Dataset):
    def __init__(self, X_data, y_data=None):
        self.X = X_data
        self.y = y_data
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        sample = self.X[idx]
        voxels = torch.from_numpy(sample['voxels']).float()
        coords = torch.from_numpy(sample['coords']).int()
        num_points = torch.from_numpy(sample['num_points']).int()
        
        target = {}
        if self.y is not None:
            boxes = self.y[idx]
            hm = np.zeros((9, *FEATURE_MAP_SIZE), dtype=np.float32)
            off = np.zeros((2, *FEATURE_MAP_SIZE), dtype=np.float32)
            z_map = np.zeros((1, *FEATURE_MAP_SIZE), dtype=np.float32)
            dim = np.zeros((3, *FEATURE_MAP_SIZE), dtype=np.float32)
            yaw = np.zeros((2, *FEATURE_MAP_SIZE), dtype=np.float32)
            mask = np.zeros((*FEATURE_MAP_SIZE,), dtype=np.float32)
            indices = np.zeros((*FEATURE_MAP_SIZE,), dtype=np.int64)

            for box in boxes:
                cls_id = CLASS_MAP.get(box['class_name'], -1)
                if cls_id == -1: continue
                fx = (box['center_x'] - PC_RANGE[0]) / VOXEL_SIZE[0] / DOWNSAMPLE
                fy = (box['center_y'] - PC_RANGE[1]) / VOXEL_SIZE[1] / DOWNSAMPLE
                if 0 <= fx < FEATURE_MAP_SIZE[1] and 0 <= fy < FEATURE_MAP_SIZE[0]:
                    ix, iy = int(fx), int(fy)
                    radius = gaussian_radius((box['length']/VOXEL_SIZE[1]/DOWNSAMPLE, box['width']/VOXEL_SIZE[0]/DOWNSAMPLE))
                    radius = max(0, int(radius))
                    draw_gaussian(hm[cls_id], (ix, iy), radius)
                    off[0, iy, ix], off[1, iy, ix] = fx - ix, fy - iy
                    z_map[0, iy, ix] = box['center_z']
                    dim[0, iy, ix] = np.log(max(1e-3, box['width']))
                    dim[1, iy, ix] = np.log(max(1e-3, box['length']))
                    dim[2, iy, ix] = np.log(max(1e-3, box['height']))
                    yaw[0, iy, ix], yaw[1, iy, ix] = np.sin(box['yaw']), np.cos(box['yaw'])
                    mask[iy, ix] = 1
                    indices[iy, ix] = iy * FEATURE_MAP_SIZE[0] + ix
            target = {'hm': hm, 'off': off, 'z': z_map, 'dim': dim, 'yaw': yaw, 'mask': mask, 'indices': indices}
        return voxels, coords, num_points, target

def collate_fn(batch):
    voxels, coords, num_points, targets = zip(*batch)
    cat_voxels = torch.cat(voxels, dim=0)
    cat_num_points = torch.cat(num_points, dim=0)
    cat_coords = []
    for i, c in enumerate(coords):
        batch_idx = torch.full((c.shape[0], 1), i, dtype=torch.int)
        cat_coords.append(torch.cat([batch_idx, c], dim=1))
    cat_coords = torch.cat(cat_coords, dim=0)
    if targets[0]:
        res_targets = {k: torch.from_numpy(np.stack([t[k] for t in targets])) for k in targets[0]}
    else:
        res_targets = {}
    return cat_voxels, cat_coords, cat_num_points, res_targets

# ===== Training Worker =====

def train_worker(rank, world_size, X_train, y_train, X_val, y_val):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    train_ds = CenterPointDataset(X_train, y_train)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=4, sampler=train_sampler, collate_fn=collate_fn, num_workers=4)
    model = CenterPointModel().cuda(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=20)
    criterion_hm = lambda pred, target: -((1 - pred)**2 * target * torch.log(pred + 1e-8) + pred**2 * (1 - target) * torch.log(1 - pred + 1e-8)).sum()
    for epoch in range(20):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        for voxels, coords, num_points, targets in train_loader:
            voxels, coords, num_points = voxels.cuda(), coords.cuda(), num_points.cuda()
            targets = {k: v.cuda() for k, v in targets.items()}
            batch_size = targets['hm'].shape[0]
            optimizer.zero_grad()
            preds = model(voxels, num_points, coords, batch_size)
            
            loss_hm = criterion_hm(preds['heatmap'], targets['hm']) / targets['mask'].sum().clamp(min=1)
            mask = targets['mask'].unsqueeze(1)
            loss_off = F.l1_loss(preds['offset'] * mask, targets['off'] * mask, reduction='sum') / mask.sum().clamp(min=1)
            loss_z = F.l1_loss(preds['z'] * mask, targets['z'] * mask, reduction='sum') / mask.sum().clamp(min=1)
            loss_dim = F.l1_loss(preds['dim'] * mask, targets['dim'] * mask, reduction='sum') / mask.sum().clamp(min=1)
            loss_yaw = F.l1_loss(preds['yaw'] * mask, targets['yaw'] * mask, reduction='sum') / mask.sum().clamp(min=1)
            
            loss = loss_hm + loss_off + loss_z + loss_dim + loss_yaw
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        if rank == 0:
            print(f"Epoch {epoch} complete. Avg loss: {total_loss/len(train_loader):.4f}")
    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(OUTPUT_DATA_PATH, "centerpoint_model.pth"))
    dist.barrier()
    dist.destroy_process_group()

# ===== Training Functions =====

def train_centerpoint(X_train: X, y_train: y, X_val: X, y_val: y, X_test: X) -> Tuple[Predictions, Predictions]:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print("Starting DDP training on 2 GPUs...")
    mp.spawn(train_worker, nprocs=2, args=(2, X_train, y_train, X_val, y_val))
    print("Inference on validation and test sets...")
    model = CenterPointModel().cuda()
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DATA_PATH, "centerpoint_model.pth")))
    model.eval()
    with open(os.path.join(OUTPUT_DATA_PATH, "data_registry.pkl"), 'rb') as f:
        registry = pickle.load(f)

    def predict_on_set(X_data):
        ds = CenterPointDataset(X_data)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for voxels, coords, num_points, _ in loader:
                voxels, coords, num_points = voxels.cuda(), coords.cuda(), num_points.cuda()
                batch_size = (coords[:, 0].max().item() + 1)
                batch_preds = model(voxels, num_points, coords, batch_size)
                hm = batch_preds['heatmap']
                hm_pool = F.max_pool2d(hm, 3, stride=1, padding=1)
                hm = hm * (hm == hm_pool).float()
                for b in range(hm.shape[0]):
                    scores, indices = torch.topk(hm[b].view(-1), 100)
                    scores, indices = scores.cpu().numpy(), indices.cpu().numpy()
                    sample_preds = []
                    token = X_data[len(all_preds)]['sample_token']
                    ego_pose = registry[token]['lidar']['ego_pose']
                    rot_mat = quat_to_rot_mat(ego_pose['rotation'])
                    ego_yaw = get_yaw_from_quat(ego_pose['rotation'])
                    for s, idx in zip(scores, indices):
                        if s < 0.1: continue
                        cls_id = idx // (FEATURE_MAP_SIZE[0] * FEATURE_MAP_SIZE[1])
                        rem = idx % (FEATURE_MAP_SIZE[0] * FEATURE_MAP_SIZE[1])
                        iy, ix = rem // FEATURE_MAP_SIZE[1], rem % FEATURE_MAP_SIZE[1]
                        dx, dy = batch_preds['offset'][b, :, iy, ix].cpu().numpy()
                        ex = (ix + dx) * DOWNSAMPLE * VOXEL_SIZE[0] + PC_RANGE[0]
                        ey = (iy + dy) * DOWNSAMPLE * VOXEL_SIZE[1] + PC_RANGE[1]
                        ez = batch_preds['z'][b, 0, iy, ix].item()
                        w, l, h = np.exp(batch_preds['dim'][b, :, iy, ix].cpu().numpy())
                        sin_y, cos_y = batch_preds['yaw'][b, :, iy, ix].cpu().numpy()
                        world_pos = rot_mat @ np.array([ex, ey, ez]) + np.array(ego_pose['translation'])
                        world_yaw = np.arctan2(sin_y, cos_y) + ego_yaw
                        sample_preds.append({
                            'center_x': float(world_pos[0]), 'center_y': float(world_pos[1]), 'center_z': float(world_pos[2]),
                            'width': float(w), 'length': float(l), 'height': float(h),
                            'yaw': float(world_yaw), 'class_name': REV_CLASS_MAP[cls_id], 'confidence': float(s)
                        })
                    all_preds.append(sample_preds)
        return all_preds

    val_preds = predict_on_set(X_val)
    test_preds = predict_on_set(X_test)
    gc.collect()
    torch.cuda.empty_cache()
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "centerpoint": train_centerpoint,
}
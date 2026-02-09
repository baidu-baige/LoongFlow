import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import pickle
import timm
from typing import Dict, List, Any
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

# Technical Specification: Global k-NN retrieval with Cosine Similarity.
# Objective: Use GPU 'torch.mm' for fast similarity between test embeddings and full gallery.

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-04/evolux/output/mlebench/hotel-id-2021-fgvc8/prepared/public"
OUTPUT_DATA_PATH = "output/6c275358-248b-46e3-a3f8-feb17fef7b7f/3/executor/output"

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

class HotelModel(nn.Module):
    """
    Backbone only for embedding extraction to avoid architecture mismatch in heads.
    """
    def __init__(self, model_name='tf_efficientnetv2_s_in21k'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        self.pooling = GeM()
        self.embedding = nn.Linear(self.backbone.num_features, 512)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pooling(feat).view(feat.size(0), -1)
        emb = self.bn(self.embedding(feat))
        return F.normalize(emb)

class HotelDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        path = self.df.loc[idx, 'image_path']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((384,384,3), dtype=np.uint8)
        return self.transform(image=img)['image']

def ensemble(
    all_val_preds: Dict[str, np.ndarray],
    all_test_preds: Dict[str, np.ndarray],
    y_val: pd.Series,
) -> pd.Series:
    print("Stage 5: Starting Global Retrieval Ensemble...")
    device = torch.device("cuda:0")
    
    # 1. Load Mappings
    map_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'hotel_id_mapping.csv'))
    id_map = dict(zip(map_df['encoded_label'], map_df['hotel_id'].astype(str)))
    id_to_enc = dict(zip(map_df['hotel_id'], map_df['encoded_label']))

    # 2. Re-infer Test Embeddings
    print("Extracting test embeddings using saved backbone...")
    test_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'sample_submission.csv'))
    test_df['image_path'] = test_df['image'].apply(lambda x: os.path.join(BASE_DATA_PATH, 'test_images', x))
    
    # Handle validation subsetting if necessary
    first_pred_shape = next(iter(all_test_preds.values())).shape[0]
    if first_pred_shape < len(test_df):
        test_df = test_df.sample(n=first_pred_shape, random_state=42).reset_index(drop=True)
    
    with open(os.path.join(OUTPUT_DATA_PATH, "val_transform.pkl"), "rb") as f:
        transform = pickle.load(f)
        
    model = HotelModel().to(device)
    sd = torch.load(os.path.join(OUTPUT_DATA_PATH, "hotel_model.pth"), map_location=device)
    # Extract only backbone/embedding weights to avoid head dimension mismatch
    backbone_sd = {k.replace('module.', ''): v for k, v in sd.items() if 'arcface' not in k and 'chain_head' not in k}
    model.load_state_dict(backbone_sd, strict=True)
    model.eval()
    
    loader = DataLoader(HotelDataset(test_df, transform), batch_size=128, shuffle=False, num_workers=8)
    test_embs = []
    with torch.no_grad():
        for imgs in loader:
            test_embs.append(model(imgs.to(device)))
    test_embs = torch.cat(test_embs, dim=0)

    # 3. Load Gallery and Align Labels
    print("Loading gallery and reconstructing label sequence...")
    gallery_embs = torch.from_numpy(np.load(os.path.join(OUTPUT_DATA_PATH, "train_gallery_embeddings.npy"))).to(device)
    gallery_embs = F.normalize(gallery_embs.float(), p=2, dim=1)
    
    full_train = pd.read_csv(os.path.join(BASE_DATA_PATH, 'train.csv'))
    full_train['image_path'] = full_train.apply(lambda r: os.path.join(BASE_DATA_PATH, 'train_images', str(r['chain']), r['image']), axis=1)
    exists = [os.path.exists(p) for p in full_train['image_path']]
    full_train = full_train[exists].reset_index(drop=True)
    
    if len(gallery_embs) < len(full_train):
        full_train = full_train.sample(n=len(gallery_embs), random_state=42).reset_index(drop=True)
        
    full_train['hotel_id_encoded'] = full_train['hotel_id'].map(id_to_enc)
    full_train['group_id'] = full_train.groupby(['hotel_id', 'timestamp']).ngroup()
    
    y_ser = full_train['hotel_id_encoded']
    groups = full_train['group_id']
    counts = y_ser.value_counts()
    single_mask = y_ser.isin(counts[counts == 1].index).values
    other_idx = np.arange(len(y_ser))[~single_mask]
    
    if len(other_idx) < 5:
        tr_idx, val_idx = next(GroupKFold(n_splits=5).split(full_train, y_ser, groups))
    else:
        y_proxy = y_ser.iloc[other_idx].copy()
        rare = y_proxy.value_counts()[lambda x: x < 5].index
        if not rare.empty:
            y_proxy[y_proxy.isin(rare)] = -1
            if 0 < (y_proxy == -1).sum() < 5:
                y_proxy[y_proxy == -1] = y_proxy[y_proxy != -1].value_counts().idxmax()
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        tr_rel, val_rel = next(sgkf.split(other_idx, y_proxy, groups.iloc[other_idx]))
        tr_idx = np.concatenate([other_idx[tr_rel], np.arange(len(y_ser))[single_mask]])
        val_idx = other_idx[val_rel]
    
    # Logic from train_and_predict: concat([train_fold, val_fold]) then rank-interleaved by DistributedSampler
    ordered_labels = np.concatenate([y_ser.iloc[tr_idx].values, y_ser.iloc[val_idx].values])
    world_size = torch.cuda.device_count()
    gallery_labels = []
    for r in range(world_size):
        gallery_labels.append(ordered_labels[r::world_size])
    gallery_labels = np.concatenate(gallery_labels)
    
    if len(gallery_labels) != len(gallery_embs):
        raise ValueError(f"Label alignment failure: {len(gallery_labels)} vs {len(gallery_embs)}")

    # 4. Fast Similarity Search (GPU)
    print("Computing global similarity matrix...")
    with torch.no_grad():
        similarities = torch.mm(test_embs, gallery_embs.t())
        _, topk_indices = torch.topk(similarities, k=min(100, len(gallery_embs)), dim=1)
        topk_indices = topk_indices.cpu().numpy()

    # 5. Format Submission
    print("Consolidating unique hotel IDs...")
    final_output = []
    for i in range(len(topk_indices)):
        unique_ids = []
        for idx in topk_indices[i]:
            hid = id_map[gallery_labels[idx]]
            if hid not in unique_ids:
                unique_ids.append(hid)
            if len(unique_ids) == 5: break
        while len(unique_ids) < 5: unique_ids.append(unique_ids[-1] if unique_ids else "0")
        final_output.append(" ".join(unique_ids))

    print(f"Ensemble complete. {len(final_output)} predictions generated.")
    return pd.Series(final_output)
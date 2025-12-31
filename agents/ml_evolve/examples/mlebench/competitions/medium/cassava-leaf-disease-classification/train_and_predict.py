# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import copy
import os
from typing import Callable, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/cassava-leaf-disease-classification/prepared/public"
OUTPUT_DATA_PATH = "output/068f0c14-e630-462e-bc46-9d2d4b1d5fc3/5/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT], Tuple[DT, DT]]


# ===== Helper Classes and Functions =====

class CassavaDataset(Dataset):
    """
    PyTorch Dataset for loading Cassava Leaf images and labels.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series = None, data_root: str = None, transforms=None):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True) if y is not None else None
        self.data_root = data_root
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int):
        img_name = self.X.iloc[index]['image_id']
        img_path = os.path.join(self.data_root, img_name)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)['image']

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        if self.y is not None:
            label = self.y.iloc[index]
            return img, torch.tensor(label, dtype=torch.long)

        return img


# Bi-Tempered Logistic Loss Helpers
def log_t(u, t):
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    if t == 1.0:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters):
    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu
    normalized_activations = normalized_activations_step_0
    for _ in range(num_iters):
        log_t_exp_t = log_t(exp_t(normalized_activations, t).sum(dim=-1, keepdim=True), t)
        normalized_activations = normalized_activations_step_0 - log_t_exp_t
    return normalized_activations


def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5):
    if label_smoothing > 0:
        num_classes = activations.shape[-1]
        labels = (1 - label_smoothing) * labels + label_smoothing / num_classes

    probabilities = compute_normalization_fixed_point(activations, t2, num_iters)
    probabilities = exp_t(probabilities, t2)

    loss_values = (-labels * log_t(probabilities, t1) -
                   (1.0 - labels) * log_t(1.0 - probabilities, t1))

    return loss_values.sum(dim=-1).mean()


# Augmentation Helpers
def mixup_data(x, y, alpha=0.4, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def tta_inference(model, loader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for images in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)

            # View 1: Original
            p1 = F.softmax(model(images), dim=1)
            # View 2: Horizontal Flip
            p2 = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
            # View 3: Vertical Flip
            p3 = F.softmax(model(torch.flip(images, dims=[2])), dim=1)
            # View 4: Transpose
            p4 = F.softmax(model(images.transpose(2, 3)), dim=1)

            avg_probs = (p1 + p2 + p3 + p4) / 4.0
            all_probs.append(avg_probs.cpu().numpy())
    if len(all_probs) == 0:
        return np.array([])
    return np.concatenate(all_probs, axis=0)


# ===== Training Functions =====

def train_efficientnet_b4_ns_heavy(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains a tf_efficientnet_b4_ns model with Bi-Tempered Loss, Mixup, CutMix and 4-view TTA.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 5
    batch_size = 16
    epochs = 10
    img_size = 512

    # Transforms
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.08, 1.0), ratio=(0.75, 1.333), p=1.0),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ToTensorV2(),
    ])

    # Datasets and Loaders
    train_ds = CassavaDataset(X_train, y_train, data_root=os.path.join(BASE_DATA_PATH, "train_images"),
                              transforms=train_transform)
    val_ds = CassavaDataset(X_val, y_val, data_root=os.path.join(BASE_DATA_PATH, "train_images"),
                            transforms=val_transform)
    test_ds = CassavaDataset(X_test, None, data_root=os.path.join(BASE_DATA_PATH, "test_images"),
                             transforms=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=num_classes)
    model.to(device)

    # Optimizer, Loss, Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=1e-6)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Training Loop
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            prob = np.random.rand()
            if prob < 0.5:
                q = np.random.rand()
                if q < 0.25:
                    # Mixup (alpha=0.4)
                    images_mix, target_a, target_b, lam = mixup_data(images, labels, alpha=0.4, device=device)
                    logits = model(images_mix)
                    one_hot_a = F.one_hot(target_a, num_classes).float()
                    one_hot_b = F.one_hot(target_b, num_classes).float()
                    mixed_labels = lam * one_hot_a + (1 - lam) * one_hot_b
                    loss = bi_tempered_logistic_loss(logits, mixed_labels, t1=0.8, t2=1.2)
                elif q < 0.50:
                    # CutMix (alpha=1.0)
                    lam = np.random.beta(1.0, 1.0)
                    rand_index = torch.randperm(images.size()[0]).to(device)
                    target_a = labels
                    target_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    logits = model(images)
                    one_hot_a = F.one_hot(target_a, num_classes).float()
                    one_hot_b = F.one_hot(target_b, num_classes).float()
                    mixed_labels = lam * one_hot_a + (1 - lam) * one_hot_b
                    loss = bi_tempered_logistic_loss(logits, mixed_labels, t1=0.8, t2=1.2)
                else:
                    # No Aug (inside 50% branch)
                    logits = model(images)
                    one_hot_labels = F.one_hot(labels, num_classes).float()
                    loss = bi_tempered_logistic_loss(logits, one_hot_labels, t1=0.8, t2=1.2)
            else:
                # No Aug (outside 50% branch)
                logits = model(images)
                one_hot_labels = F.one_hot(labels, num_classes).float()
                loss = bi_tempered_logistic_loss(logits, one_hot_labels, t1=0.8, t2=1.2)

            loss.backward()
            optimizer.step()

        # Validation for best weights selection
        model.eval()
        val_preds_epoch = []
        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                val_logits = model(val_images)
                val_preds_epoch.append(F.softmax(val_logits, dim=1).cpu().numpy())
        val_preds_epoch = np.concatenate(val_preds_epoch, axis=0)
        acc = (val_preds_epoch.argmax(axis=1) == y_val.values).mean()

        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Perform 4-view TTA Inference
    val_predictions = tta_inference(model, val_loader, device)
    test_predictions = tta_inference(model, test_loader, device)

    return val_predictions, test_predictions


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "efficientnet_b4_ns_heavy": train_efficientnet_b4_ns_heavy,
}

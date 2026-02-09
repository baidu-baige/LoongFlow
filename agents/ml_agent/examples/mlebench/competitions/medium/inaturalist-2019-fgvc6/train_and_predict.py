import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Any, Dict, Callable

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/inaturalist-2019-fgvc6/prepared/public"
OUTPUT_DATA_PATH = "output/57a5d42d-daf8-4241-8197-424dd36c00a6/1/executor/output"

# Task-adaptive concrete type definitions
X = Any      # INatDataset (torch.utils.data.Dataset)
y = Any     # pd.Series containing category_id labels
Predictions = Any     # np.ndarray of logits

# Prediction Function Type definition
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class ConvNeXtAux(nn.Module):
    """
    ConvNeXt-Base model with auxiliary heads for taxonomic supervision (Family and Order).
    """
    def __init__(self, num_species: int, num_families: int, num_orders: int):
        super().__init__()
        # Load ConvNeXt-Base backbone pre-trained on ImageNet-22k
        self.backbone = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=0)
        
        # ConvNeXt-Base feature dimension is 1024
        feature_dim = 1024
        
        # Primary head for species classification
        self.fc_species = nn.Linear(feature_dim, num_species)
        
        # Auxiliary heads for taxonomic regularization
        self.fc_family = nn.Linear(feature_dim, num_families)
        self.fc_order = nn.Linear(feature_dim, num_orders)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        species_logits = self.fc_species(features)
        family_logits = self.fc_family(features)
        order_logits = self.fc_order(features)
        return species_logits, family_logits, order_logits

def train_convnext_aux(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a ConvNeXt-Base model with auxiliary taxonomic heads.
    """
    # 1. Prepare Label Mappings for taxonomic supervision
    # We use the metadata stored in the INatDataset objects during preprocessing
    train_meta = X_train.metadata
    val_meta = X_val.metadata
    full_meta = pd.concat([train_meta, val_meta], axis=0)
    
    # Species label encoding (mapping category_id to 0...1009)
    all_cat_ids = sorted(pd.concat([y_train, y_val]).unique())
    cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(all_cat_ids)}
    num_species = len(all_cat_ids)
    
    # Family and Order label encoding
    family_le = LabelEncoder()
    order_le = LabelEncoder()
    family_le.fit(full_meta['family'])
    order_le.fit(full_meta['order'])
    num_families = len(family_le.classes_)
    num_orders = len(order_le.classes_)
    
    # Create taxonomy lookup (species_idx -> family_idx and order_idx)
    # We map the category_id from the targets to the metadata
    full_meta['target_species'] = pd.concat([y_train, y_val]).values
    taxonomy_map = full_meta.groupby('target_species').first()[['family', 'order']]
    
    spec_to_fam = np.zeros(num_species, dtype=np.int64)
    spec_to_ord = np.zeros(num_species, dtype=np.int64)
    
    for cat_id, row in taxonomy_map.iterrows():
        if cat_id in cat_to_idx:
            s_idx = cat_to_idx[cat_id]
            spec_to_fam[s_idx] = family_le.transform([row['family']])[0]
            spec_to_ord[s_idx] = order_le.transform([row['order']])[0]
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec_to_fam_tensor = torch.from_numpy(spec_to_fam).to(device)
    spec_to_ord_tensor = torch.from_numpy(spec_to_ord).to(device)
    
    # 2. Update Dataset Labels to dense indices
    X_train.labels = np.array([cat_to_idx[c] for c in y_train.values])
    X_val.labels = np.array([cat_to_idx[c] for c in y_val.values])
    
    # 3. Model, Optimizer, and DataLoaders
    model = ConvNeXtAux(num_species, num_families, num_orders).to(device)
    
    batch_size = 32
    num_workers = 8
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(X_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    
    # Hyperparameters
    epochs = 30
    warmup_epochs = 5
    
    # Adjust for validation mode (small datasets)
    if len(X_train) < 1000:
        epochs = 2
        warmup_epochs = 1
        
    # Scheduler: Linear Warmup followed by Cosine Annealing
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    
    # Loss: CrossEntropy with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                s_logits, f_logits, o_logits = model(images)
                
                # Primary Species Loss
                loss_species = criterion(s_logits, labels)
                
                # Auxiliary Taxonomic Losses
                f_targets = spec_to_fam_tensor[labels]
                o_targets = spec_to_ord_tensor[labels]
                loss_family = criterion(f_logits, f_targets)
                loss_order = criterion(o_logits, o_targets)
                
                # Combined Loss (Auxiliary weight = 0.2)
                loss = loss_species + 0.2 * loss_family + 0.2 * loss_order
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        
    # 5. Inference
    model.eval()
    val_preds = []
    test_preds = []
    
    with torch.no_grad():
        # Validation Set
        for images, _ in val_loader:
            images = images.to(device)
            with autocast():
                s_logits, _, _ = model(images)
            val_preds.append(s_logits.cpu().numpy())
            
        # Test Set
        for images in test_loader:
            images = images.to(device)
            with autocast():
                s_logits, _, _ = model(images)
            test_preds.append(s_logits.cpu().numpy())
            
    val_outputs = np.concatenate(val_preds, axis=0).astype(np.float32)
    test_outputs = np.concatenate(test_preds, axis=0).astype(np.float32)
    
    # Replace any potential NaNs/Infs (safety check)
    val_outputs = np.nan_to_num(val_outputs, nan=0.0, posinf=1e6, neginf=-1e6)
    test_outputs = np.nan_to_num(test_outputs, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return val_outputs, test_outputs

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "convnext_aux": train_convnext_aux,
}
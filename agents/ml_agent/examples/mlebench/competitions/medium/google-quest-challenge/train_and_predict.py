import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import RobertaModel, RobertaConfig, BertModel, BertConfig, get_linear_schedule_with_warmup
from typing import Tuple, Dict, Callable

# Constants and Paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/aaa741b3-cb02-44fc-a666-dd434e563444/8/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.DataFrame
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Reproducibility =====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ===== Dataset Implementation =====

class QuestDataset(Dataset):
    """Dataset for Google QUEST dual-model tokens and metadata."""
    def __init__(self, X_df: pd.DataFrame, y_df: pd.DataFrame = None, model_type: str = 'roberta'):
        self.X = X_df.values.astype(np.float32)
        self.y = y_df.values.astype(np.float32) if y_df is not None else None
        self.model_type = model_type
        
        # Mapping based on preprocess.py column structure:
        # r_id (0:512), r_mask (512:1024), b_id (1024:1536), b_mask (1536:2048),
        # category_id (2048), host_id (2049), scalars (2050:2067)
        if model_type == 'roberta':
            self.id_slice = slice(0, 512)
            self.mask_slice = slice(512, 1024)
        else: # bert
            self.id_slice = slice(1024, 1536)
            self.mask_slice = slice(1536, 2048)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X[idx]
        input_ids = torch.as_tensor(row[self.id_slice], dtype=torch.long)
        attention_mask = torch.as_tensor(row[self.mask_slice], dtype=torch.long)
        category_id = torch.as_tensor(int(row[2048]), dtype=torch.long)
        host_id = torch.as_tensor(int(row[2049]), dtype=torch.long)
        scalar_features = torch.as_tensor(row[2050:2067], dtype=torch.float32)
        
        if self.y is not None:
            targets = torch.as_tensor(self.y[idx], dtype=torch.float32)
            return input_ids, attention_mask, category_id, host_id, scalar_features, targets
        return input_ids, attention_mask, category_id, host_id, scalar_features

# ===== High-Stability Model Implementation =====

class StableQuestModel(nn.Module):
    """
    Dual-backbone compatible model with CLS+Mean pooling and linear head.
    Supports RoBERTa and BERT backbones with multi-sample dropout.
    """
    def __init__(self, model_path: str, model_type: str, n_cats: int, n_hosts: int):
        super(StableQuestModel, self).__init__()
        self.model_type = model_type
        
        local_files_only = os.path.exists(model_path)
        if model_type == 'roberta':
            config = RobertaConfig.from_pretrained(model_path, local_files_only=local_files_only)
            self.backbone = RobertaModel.from_pretrained(model_path, config=config, local_files_only=local_files_only)
        else: # bert
            config = BertConfig.from_pretrained(model_path, local_files_only=local_files_only)
            self.backbone = BertModel.from_pretrained(model_path, config=config, local_files_only=local_files_only)
            
        # Meta-embeddings (dim=4 as per specification)
        self.cat_emb = nn.Embedding(n_cats + 5, 4)
        self.host_emb = nn.Embedding(n_hosts + 5, 4)
        
        # Dimension: CLS(768) + Mean(768) + Cat(4) + Host(4) + Scalars(17) = 1561
        input_dim = 768 * 2 + 4 + 4 + 17
        
        # Linear head with 5-mask Multi-sample Dropout (0.5)
        self.output_head = nn.Linear(input_dim, 30)
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def forward(self, input_ids, attention_mask, category_id, host_id, scalar_features):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        cls_pool = last_hidden[:, 0, :]
        mean_pool = torch.mean(last_hidden, dim=1)
        
        c_emb = self.cat_emb(category_id)
        h_emb = self.host_emb(host_id)
        
        concat = torch.cat([cls_pool, mean_pool, c_emb, h_emb, scalar_features], dim=1)
        
        logits = torch.mean(torch.stack([
            self.output_head(dropout(concat)) for dropout in self.dropouts
        ], dim=0), dim=0)
        
        return logits

# ===== Training Logic =====

def train_single_backbone(
    model_type: str,
    X_train: X,
    y_train: y,
    X_val: X,
    X_test: X,
    device: torch.device
) -> Tuple[Predictions, Predictions]:
    """Helper to train one transformer backbone and generate predictions."""
    print(f"Training {model_type} backbone...")
    
    # Path configuration
    base_paths = {
        'roberta': '/mnt/pfs/loongflow/devmachine/2-05/models/roberta-base',
        'bert': '/mnt/pfs/loongflow/devmachine/2-05/models/bert-base-uncased'
    }
    model_path = base_paths[model_type] if os.path.exists(base_paths[model_type]) else f"{model_type}-base-uncased" if model_type=='bert' else "roberta-base"
    
    n_cats = int(max(X_train['category_id'].max(), X_test['category_id'].max())) + 1
    n_hosts = int(max(X_train['host_id'].max(), X_test['host_id'].max())) + 1
    
    model = StableQuestModel(model_path, model_type, n_cats, n_hosts).to(device)
    
    # Differential LRs: Backbone 2e-5, Head 1e-3
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 2e-5},
        {'params': model.cat_emb.parameters(), 'lr': 1e-3},
        {'params': model.host_emb.parameters(), 'lr': 1e-3},
        {'params': model.output_head.parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)
    
    train_loader = DataLoader(QuestDataset(X_train, y_train, model_type), batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(QuestDataset(X_val, model_type=model_type), batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(QuestDataset(X_test, model_type=model_type), batch_size=16, shuffle=False, num_workers=4)
    
    epochs = 4
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = [t.to(device) for t in batch]
            ids, mask, cat, host, lens, targets = batch
            
            optimizer.zero_grad()
            with autocast():
                logits = model(ids, mask, cat, host, lens)
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
    def predict(loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = [t.to(device) for t in batch]
                ids, mask, cat, host, lens = batch
                with autocast():
                    logits = model(ids, mask, cat, host, lens)
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(preds)
    
    val_preds = predict(val_loader)
    test_preds = predict(test_loader)
    
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_preds, test_preds

def train_dual_transformer(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Executes sequential fine-tuning of RoBERTa and BERT.
    Averages predictions for high-stability multi-label regression.
    """
    print("Execution Stage: train_dual_transformer")
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Train RoBERTa-base
    r_val, r_test = train_single_backbone('roberta', X_train, y_train, X_val, X_test, device)
    
    # 2. Train BERT-base-uncased
    b_val, b_test = train_single_backbone('bert', X_train, y_train, X_val, X_test, device)
    
    # 3. Ensemble (Simple Average)
    val_preds = (r_val + b_val) / 2.0
    test_preds = (r_test + b_test) / 2.0
    
    # Validation against NaNs
    if np.isnan(val_preds).any() or np.isnan(test_preds).any():
        raise ValueError("Predictions contain NaNs. Check learning rates and normalization.")
        
    print("Dual-transformer training complete. Predictions generated.")
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "dual_transformer": train_dual_transformer,
}
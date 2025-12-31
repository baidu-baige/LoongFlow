# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import gc
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2Config, DebertaV2Model, get_cosine_schedule_with_warmup

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/google-quest-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/ea4f9f02-c6be-4ad0-a6fd-71930e8fb81e/15/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT], Tuple[DT, DT]]


# ===== Helper Functions =====

def compute_spearman(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's correlation coefficient.
    """
    scores = []
    for i in range(y_true.shape[1]):
        score, _ = spearmanr(y_true[:, i], y_pred[:, i])
        if np.isnan(score):
            score = 0.0
        scores.append(score)
    return np.mean(scores)


class QuestDataset(Dataset):
    """
    Dataset for Google QUEST challenge handling text tokens and metadata.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame = None):
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        else:
            self.X = X

        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                self.y = y.values
            else:
                self.y = y
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.X[idx, :512], dtype=torch.long),
            'attention_mask': torch.tensor(self.X[idx, 512:1024], dtype=torch.long),
            'meta': torch.tensor(self.X[idx, 1024:], dtype=torch.float)
        }
        if self.y is not None:
            item['targets'] = torch.tensor(self.y[idx], dtype=torch.float)
        return item


# ===== DeBERTa Model =====

class DebertaV3LargeSwaModel(nn.Module):
    """
    DeBERTa-v3-large with Hybrid Pooling and Multi-Sample Dropout.
    Hybrid Pooling: Last 4 CLS tokens + Global Average Pooling of last layer.
    """

    def __init__(self, meta_dim: int):
        super().__init__()
        # Use use_safetensors=True to address torch.load vulnerability restriction in the environment
        self.backbone = DebertaV2Model.from_pretrained(
            'microsoft/deberta-v3-large',
            output_hidden_states=True,
            use_safetensors=True
        )
        # 4 * 1024 (last 4 CLS) + 1024 (avg pooling) = 5120
        self.fc1 = nn.Linear(5120 + meta_dim, 1024)
        self.mish = nn.Mish()
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        self.fc2 = nn.Linear(1024, 30)

    def forward(self, input_ids, attention_mask, meta):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # Last 4 layers CLS tokens: (bs, 4 * 1024)
        cls_tokens = torch.cat([hidden_states[-i][:, 0, :] for i in range(1, 5)], dim=1)

        # Global average pooling of the last layer: (bs, 1024)
        avg_pool = torch.mean(hidden_states[-1], dim=1)

        # Hybrid concatenation
        x = torch.cat([cls_tokens, avg_pool, meta], dim=1)

        x = self.fc1(x)
        x = self.mish(x)

        # Multi-sample dropout (5 passes)
        logits = torch.mean(torch.stack([self.fc2(dropout(x)) for dropout in self.dropouts]), dim=0)
        return logits


# ===== Training Function =====

def train_deberta_v3_large_swa(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains DeBERTa-v3-large with SWA and Hybrid Pooling.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_dim = X_train.shape[1] - 1024

    # Hyperparameters from Strategy Plan
    epochs = 8
    swa_start_epoch = 4  # Start at Epoch 4 of 8 (index 3)
    batch_size = 4
    accumulation_steps = 4  # Effective BS = 16
    backbone_lr = 4e-6
    head_lr = 2e-5

    train_loader = DataLoader(QuestDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(QuestDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(QuestDataset(X_test), batch_size=batch_size, shuffle=False)

    model = DebertaV3LargeSwaModel(meta_dim).to(device)
    swa_model = torch.optim.swa_utils.AveragedModel(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': head_lr}
    ])

    num_train_steps = (len(train_loader) // accumulation_steps) * epochs
    num_warmup_steps = int(num_train_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=head_lr)

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            meta = batch['meta'].to(device)
            targets = batch['targets'].to(device)

            with torch.cuda.amp.autocast():
                logits = model(ids, mask, meta)
                loss = criterion(logits, targets) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if epoch + 1 >= swa_start_epoch:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()

    # Update BatchNorm (though DeBERTa uses LayerNorm)
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    swa_model.eval()
    val_preds, test_preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids, mask, meta = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['meta'].to(
                device)
            with torch.cuda.amp.autocast():
                pred = torch.sigmoid(swa_model(ids, mask, meta))
            val_preds.append(pred.cpu().numpy())

        for batch in test_loader:
            ids, mask, meta = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['meta'].to(
                device)
            with torch.cuda.amp.autocast():
                pred = torch.sigmoid(swa_model(ids, mask, meta))
            test_preds.append(pred.cpu().numpy())

    val_preds = np.vstack(val_preds)
    test_preds = np.vstack(test_preds)

    # Cleanup
    del model, swa_model, optimizer, scheduler, swa_scheduler, train_loader
    gc.collect()
    torch.cuda.empty_cache()

    return val_preds, test_preds


# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "deberta_v3_large_swa": train_deberta_v3_large_swa,
}

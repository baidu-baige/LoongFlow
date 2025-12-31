# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE_DATA_PATH = "/root/workspace/evolux_ml/output/mlebench/stanford-covid-vaccine/prepared/public"
OUTPUT_DATA_PATH = "output/832d0196-b83e-4fa9-8ea2-3588ff903a43/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT, Any], Tuple[DT, DT]]


# ===== Helper Classes and Functions for GCN =====

class RNADataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame = None):
        self.X = X
        self.y = y
        self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        nf = torch.tensor(row['node_features'], dtype=torch.float32)
        adj = torch.tensor(row['adj_matrix'], dtype=torch.float32)
        sn_filter = torch.tensor(row['SN_filter'], dtype=torch.float32)
        seq_scored = torch.tensor(row['seq_scored'], dtype=torch.long)
        seq_len = torch.tensor(row['seq_length'], dtype=torch.long)

        if self.y is not None:
            y_row = self.y.iloc[idx]
            # Targets are lists of length 'seq_scored'
            targets_list = [y_row[col] for col in self.target_cols]
            targets_np = np.stack(targets_list, axis=1)  # (seq_scored, 5)
            # Pad targets to full sequence length for easier batching
            full_targets = np.zeros((nf.shape[0], 5), dtype=np.float32)
            full_targets[:targets_np.shape[0], :] = targets_np
            return nf, adj, torch.tensor(full_targets, dtype=torch.float32), sn_filter, seq_scored, seq_len
        else:
            # For test, return zero targets as placeholder
            return nf, adj, torch.zeros((nf.shape[0], 5), dtype=torch.float32), sn_filter, torch.tensor(0,
                                                                                                        dtype=torch.long), seq_len


def collate_fn(batch):
    nfs, adjs, targets, sns, scored_lens, seq_lens = zip(*batch)
    max_len = max([x.shape[0] for x in nfs])

    padded_nfs = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in nfs])
    padded_adjs = torch.stack([F.pad(x, (0, max_len - x.shape[1], 0, max_len - x.shape[0])) for x in adjs])
    padded_targets = torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in targets])

    return padded_nfs, padded_adjs, padded_targets, torch.stack(sns), torch.stack(scored_lens), torch.stack(seq_lens)


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, L, Fin), adj: (B, L, L)
        x = self.linear(x)
        return torch.matmul(adj, x)


class GCNModel(nn.Module):
    def __init__(self, input_dim: int = 14, hidden_dim: int = 128, output_dim: int = 5, n_layers: int = 4):
        super(GCNModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            h = layer(x, adj)
            h = F.relu(h)
            x = x + h  # Residual connection
        return self.readout(x)


def mcrmse_loss(pred, target, scored_lens, weights):
    """
    Computes weighted MCRMSE loss.
    pred, target: (B, L, 5)
    scored_lens: (B,)
    weights: (B,)
    """
    B, L, C = pred.shape
    device = pred.device

    # Create mask for scored positions
    mask = torch.arange(L, device=device).expand(B, L) < scored_lens.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand(B, L, C)  # (B, L, 5)

    # MSE per position and target
    mse = (pred - target) ** 2
    mse = mse * mask.float()

    # Mean MSE per sample and target
    mse_sum = mse.sum(dim=1)  # (B, 5)
    # Avoid division by zero for test samples in val loader if any
    safe_scored_lens = torch.clamp(scored_lens, min=1)
    mse_mean = mse_sum / safe_scored_lens.unsqueeze(1).float()

    # RMSE per sample and target
    rmse = torch.sqrt(mse_mean + 1e-8)  # (B, 5)

    # MCRMSE per sample
    mcrmse_per_sample = rmse.mean(dim=1)  # (B,)

    # Weighted average over batch
    return (mcrmse_per_sample * weights).mean()


# ===== Training Functions =====

def train_gcn(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trains a GCN model and returns predictions for validation and test sets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    epochs = 100
    lr = 0.001

    # Prepare DataLoaders
    train_ds = RNADataset(X_train, y_train)
    val_ds = RNADataset(X_val, y_val)
    test_ds = RNADataset(X_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize Model, Optimizer, Scheduler
    model = GCNModel(input_dim=14, hidden_dim=128, output_dim=5, n_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for nfs, adjs, targets, sns, scored_lens, seq_lens in train_loader:
            nfs, adjs, targets, sns, scored_lens = nfs.to(device), adjs.to(device), targets.to(device), sns.to(
                device), scored_lens.to(device)

            optimizer.zero_grad()
            preds = model(nfs, adjs)

            # Weighted loss: 2x for SN_filter == 1
            weights = 1.0 + sns
            loss = mcrmse_loss(preds, targets, scored_lens, weights)

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation for early feedback (optional logging)
        model.eval()
        with torch.no_grad():
            pass  # Validation logic can be added here if needed

    # Generate predictions
    def get_predictions(loader, ids):
        model.eval()
        all_preds = []
        all_seq_lens = []
        with torch.no_grad():
            for batch in loader:
                nfs, adjs, _, _, _, seq_lens = batch
                nfs, adjs = nfs.to(device), adjs.to(device)
                preds = model(nfs, adjs).cpu().numpy()
                for i in range(len(preds)):
                    length = seq_lens[i].item()  # Convert tensor to Python int for slicing
                    all_preds.append(preds[i, :length, :])

        target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        formatted_rows = []
        for i, p in enumerate(all_preds):
            row = {'id': ids.iloc[i]}
            for j, col in enumerate(target_cols):
                row[col] = p[:, j].tolist()
            formatted_rows.append(row)
        return pd.DataFrame(formatted_rows)

    val_predictions = get_predictions(val_loader, X_val['id'])
    test_predictions = get_predictions(test_loader, X_test['id'])

    return val_predictions, test_predictions


# ===== Model Registry =====
# Register ALL training functions here for the pipeline to use
# Key: Descriptive model name (e.g., "lgbm_tuned", "neural_net")
# Value: The training function reference

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "gcn_base": train_gcn,
}

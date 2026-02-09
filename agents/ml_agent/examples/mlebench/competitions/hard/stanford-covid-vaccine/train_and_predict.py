import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple, Any, Dict, Callable

# Task-adaptive type definitions
X = np.ndarray           # Feature matrix: (N, 107, 111)
y = np.ndarray           # Target vector: (N, 107, 5)
Predictions = np.ndarray # Predictions: (N, 107, 5)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class MCRMSELoss(nn.Module):
    """Mean Columnwise Root Mean Squared Error Loss calculated on the scored positions."""
    def __init__(self, scored_len: int = 68):
        super(MCRMSELoss, self).__init__()
        self.scored_len = scored_len

    def forward(self, y_pred, y_true):
        # Slice to scored length: (N, 68, 5)
        y_pred = y_pred[:, :self.scored_len, :]
        y_true = y_true[:, :self.scored_len, :]
        
        # Calculate RMSE per column
        mse = torch.mean((y_pred - y_true) ** 2, dim=1) # (N, 5)
        rmse = torch.sqrt(mse + 1e-8) # (N, 5)
        return torch.mean(rmse) # Scalar mean over samples and columns

class GNNLayer(nn.Module):
    """Simple Graph Convolution Layer with LayerNorm and Residual Connection."""
    def __init__(self, in_channels: int, out_channels: int):
        super(GNNLayer, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, L, C), adj: (N, L, L)
        # Graph convolution: out = ReLU(A * X * W)
        support = self.lin(x) # (N, L, out_channels)
        out = torch.matmul(adj, support) # (N, L, out_channels)
        out = self.norm(out)
        out = self.activation(out)
        return out + x if x.shape == out.shape else out

class RNAGNN(nn.Module):
    def __init__(self, embed_dim_seq=64, embed_dim_struct=32, embed_dim_loop=32):
        super(RNAGNN, self).__init__()
        # Embeddings for categorical features
        self.emb_seq = nn.Embedding(4, embed_dim_seq)
        self.emb_struct = nn.Embedding(3, embed_dim_struct)
        self.emb_loop = nn.Embedding(7, embed_dim_loop)
        
        # Spatial Dropout (Dropout2d applied to channels)
        self.spatial_dropout = nn.Dropout2d(0.2)
        
        # Input projection to GNN dimension
        node_dim = embed_dim_seq + embed_dim_struct + embed_dim_loop + 1 # +1 for normalized pos
        self.proj = nn.Linear(node_dim, 128)
        
        # Stacked GNN Layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(128, 128) for _ in range(4)
        ])
        
        # Multi-target regression head
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 5)
        )

    def forward(self, x_node: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x_node: (N, 107, 4) -> [seq_idx, struct_idx, loop_idx, pos]
        seq = self.emb_seq(x_node[:, :, 0].long())
        struct = self.emb_struct(x_node[:, :, 1].long())
        loop = self.emb_loop(x_node[:, :, 2].long())
        pos = x_node[:, :, 3:4]
        
        # Concatenate features along the last dimension
        x = torch.cat([seq, struct, loop, pos], dim=-1) # (N, 107, node_dim)
        
        # Apply Spatial Dropout: (N, L, C) -> (N, C, L) -> Dropout -> (N, L, C)
        x = x.transpose(1, 2)
        x = self.spatial_dropout(x)
        x = x.transpose(1, 2)
        
        x = self.proj(x)
        
        # Pass through GNN layers
        for layer in self.gnn_layers:
            x = layer(x, adj)
        
        # Position-wise output
        return self.head(x)

def train_gnn_model(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a Graph Neural Network on RNA sequences and predicts 5 degradation values per base.
    """
    # 1. Device configuration
    # Utilizing the first H20 GPU. For this small dataset, multi-GPU overhead exceeds benefits.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training GNN on device: {device}")

    # 2. Data Preparation and Adjacency Normalization
    def prepare_data(X_arr):
        # Node features are the first 4 dimensions, Adjacency is the rest (107x107)
        node_feats = X_arr[:, :, :4]
        adj = X_arr[:, :, 4:]
        # Add self-loops to Adjacency
        adj = adj + np.eye(107)[None, :, :]
        return torch.tensor(node_feats, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32)

    X_train_node, X_train_adj = prepare_data(X_train)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    
    X_val_node, X_val_adj = prepare_data(X_val)
    y_val_torch = torch.tensor(y_val, dtype=torch.float32)
    
    X_test_node, X_test_adj = prepare_data(X_test)

    # Use DataLoaders for efficient batching
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train_node, X_train_adj, y_train_torch), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_node, X_val_adj, y_val_torch), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_node, X_test_adj), batch_size=batch_size, shuffle=False)

    # 3. Model, Optimizer, and Loss Setup
    model = RNAGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = MCRMSELoss(scored_len=68)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 4. Training Loop
    epochs = 50
    best_val_loss = float('inf')
    best_model_state = None

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for b_node, b_adj, b_y in train_loader:
            b_node, b_adj, b_y = b_node.to(device), b_adj.to(device), b_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(b_node, b_adj)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_node, b_adj, b_y in val_loader:
                b_node, b_adj, b_y = b_node.to(device), b_adj.to(device), b_y.to(device)
                outputs = model(b_node, b_adj)
                v_loss = criterion(outputs, b_y)
                val_loss += v_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Train MCRMSE: {avg_train_loss:.4f} | Val MCRMSE: {avg_val_loss:.4f}")

    # 5. Inference
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    
    def predict(loader):
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                # loaders yield (node, adj, [y])
                b_node, b_adj = batch[0].to(device), batch[1].to(device)
                preds = model(b_node, b_adj)
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    val_preds = predict(val_loader)
    test_preds = predict(test_loader)

    print(f"Training complete. Best Val MCRMSE: {best_val_loss:.4f}")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "gnn_rna": train_gnn_model,
}
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Tuple, Any, Dict, Callable
import os

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/detecting-insults-in-social-commentary/prepared/public"
OUTPUT_DATA_PATH = "output/f0abd8d6-b251-4e86-b7b3-c7603506ee1b/1/executor/output"

# Task-adaptive type definitions
X = np.ndarray  # Preprocessed dense feature matrix
y = pd.Series  # Binary target labels
Predictions = np.ndarray  # Probability scores [0, 1]

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]


# ===== DeBERTa Helper Classes =====

class InsultDataset(Dataset):
    def __init__(self, ids: np.ndarray, masks: np.ndarray, targets: pd.Series = None):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.long)
        self.targets = torch.tensor(targets.values, dtype=torch.float) if targets is not None else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.ids[idx],
            'attention_mask': self.masks[idx]
        }
        if self.targets is not None:
            item['labels'] = self.targets[idx]
        return item


class DebertaMultiSampleModel(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.config = self.deberta.config
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the first token [CLS] for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Multi-sample dropout: apply dropout multiple times and average logits
        # This helps regularization on small datasets.
        logits = torch.mean(torch.stack([
            self.classifier(self.dropout(pooled_output)) for _ in range(5)
        ]), dim=0)
        return logits.squeeze(-1)


# ===== Training Functions =====

def train_deberta_ridge_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains an ensemble of Fine-tuned DeBERTa-v3-base and Calibrated Ridge Classifier.

    This implementation handles dynamic feature indexing to avoid slicing errors
    caused by TF-IDF vocabulary size variations (e.g., in validation mode).
    """
    print("Stage 4: Starting training for DeBERTa-v3 + Ridge Ensemble...")

    # Dynamic Slicing Logic
    # Structure from preprocess: [TF-IDF (N), Dates (3), IDs (256), Masks (256)]
    total_cols = X_train.shape[1]
    SEQ_LEN = 256
    DATE_FEATS = 3

    mask_start = total_cols - SEQ_LEN
    ids_start = mask_start - SEQ_LEN
    date_start = ids_start - DATE_FEATS

    print(f"Feature Map: TF-IDF [0:{date_start}], Dates [{date_start}:{ids_start}], "
          f"IDs [{ids_start}:{mask_start}], Masks [{mask_start}:{total_cols}]")

    # 1. Prepare Data for Ridge (TF-IDF + Date Features)
    X_train_ridge = X_train[:, :ids_start]
    X_val_ridge = X_val[:, :ids_start]
    X_test_ridge = X_test[:, :ids_start]

    # 2. Prepare Data for DeBERTa (IDs + Masks)
    # Ensure IDs and Masks are converted back to long integers for embeddings
    train_ids = X_train[:, ids_start:mask_start].astype(np.int64)
    train_masks = X_train[:, mask_start:].astype(np.int64)

    val_ids = X_val[:, ids_start:mask_start].astype(np.int64)
    val_masks = X_val[:, mask_start:].astype(np.int64)

    test_ids = X_test[:, ids_start:mask_start].astype(np.int64)
    test_masks = X_test[:, mask_start:].astype(np.int64)

    # Sanity check on transformer input shapes
    if train_ids.shape[1] != SEQ_LEN or train_masks.shape[1] != SEQ_LEN:
        raise RuntimeError(
            f"Transformer input slice error: IDs shape {train_ids.shape}, Masks shape {train_masks.shape}")

    # --- Part A: Ridge Training ---
    print("Training Calibrated Ridge Classifier...")
    # Using Ridge for high-dimensional TF-IDF efficiency
    base_ridge = RidgeClassifier(alpha=1.0, random_state=42)
    clf_ridge = CalibratedClassifierCV(base_ridge, cv=5, method='sigmoid')
    clf_ridge.fit(X_train_ridge, y_train)

    ridge_val_probs = clf_ridge.predict_proba(X_val_ridge)[:, 1]
    ridge_test_probs = clf_ridge.predict_proba(X_test_ridge)[:, 1]

    # --- Part B: DeBERTa Training ---
    print("Training DeBERTa-v3-base...")
    # Use single GPU as the dataset size (~4k) is too small for DDP overhead to be beneficial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = InsultDataset(train_ids, train_masks, y_train)
    val_dataset = InsultDataset(val_ids, val_masks)
    test_dataset = InsultDataset(test_ids, test_masks)

    # Batch size 16 as per spec
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = DebertaMultiSampleModel().to(device)

    epochs = 4
    lr = 2e-5
    weight_decay = 0.01

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_train_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {total_loss / len(train_loader):.4f}")

    # Inference helper
    def get_deberta_preds(loader):
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
        return np.array(all_preds)

    deberta_val_probs = get_deberta_preds(val_loader)
    deberta_test_probs = get_deberta_preds(test_loader)

    # --- Part C: Ensemble ---
    # Weighted ensemble: DeBERTa usually outperforms Ridge on AUC, but Ridge captures
    # exact keyword matches that help in social commentary.
    val_preds = 0.6 * deberta_val_probs + 0.4 * ridge_val_probs
    test_preds = 0.6 * deberta_test_probs + 0.4 * ridge_test_probs

    # Check for invalid values
    if np.isnan(val_preds).any() or np.isnan(test_preds).any():
        val_preds = np.nan_to_num(val_preds, nan=0.0)
        test_preds = np.nan_to_num(test_preds, nan=0.0)

    print(f"Training complete. Returning predictions for {len(val_preds)} val and {len(test_preds)} test samples.")
    return val_preds, test_preds


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "deberta_ridge_ensemble": train_deberta_ridge_ensemble,
}
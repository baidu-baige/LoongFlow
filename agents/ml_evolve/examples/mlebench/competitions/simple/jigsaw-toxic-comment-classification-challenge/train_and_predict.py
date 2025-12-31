# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/40d01db3-cd9d-46e2-8d9c-d192fb8addff/1/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT], Tuple[DT, DT]]


# ===== Training Functions =====

def train_bert_transformer(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains a BERT-base-uncased model for multi-label toxicity classification.
    Optimizes Mean ROC AUC across 6 target categories.
    """
    # Step 1: Configuration and Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    MAX_LEN = 256
    BATCH_SIZE = 32
    EPOCHS = 3
    LR = 2e-5
    WEIGHT_DECAY = 0.01

    # Dataset Class for PyTorch
    class JigsawDataset(Dataset):
        def __init__(self, ids, masks, labels=None):
            self.ids = ids
            self.masks = masks
            self.labels = labels

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            item = {
                'input_ids': torch.tensor(self.ids[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.masks[idx], dtype=torch.long)
            }
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

    def get_data_arrays(df):
        # Extract input_ids and attention_mask from the feature engineered DataFrame
        # create_features stacked them horizontally: [ids...][masks...]
        vals = df.values
        ids = vals[:, :MAX_LEN].astype(np.int64)
        masks = vals[:, MAX_LEN:].astype(np.int64)
        return ids, masks

    # Ensure inputs are treated as DataFrames to extract arrays correctly
    X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    X_val_df = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val
    X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
    y_train_df = pd.DataFrame(y_train) if not isinstance(y_train, pd.DataFrame) else y_train
    y_val_df = pd.DataFrame(y_val) if not isinstance(y_val, pd.DataFrame) else y_val

    # Step 2: Prepare DataLoaders
    train_ids, train_masks = get_data_arrays(X_train_df)
    val_ids, val_masks = get_data_arrays(X_val_df)
    test_ids, test_masks = get_data_arrays(X_test_df)

    train_dataset = JigsawDataset(train_ids, train_masks, y_train_df.values)
    val_dataset = JigsawDataset(val_ids, val_masks, y_val_df.values)
    test_dataset = JigsawDataset(test_ids, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 3: Build and Configure Model
    # BERT-base-uncased with 6-unit linear head for multi-label classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
    model.to(device)

    # Optimizer from torch.optim as AdamW was moved/deprecated in transformers
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # BCEWithLogitsLoss handles multi-label natively with logits
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_state = None

    # Step 4: Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(ids, attention_mask=masks)
            logits = outputs.logits
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation for Checkpointing based on Mean ROC AUC
        model.eval()
        val_epoch_preds = []
        val_epoch_targets = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                labels = batch['labels'].numpy()

                logits = model(ids, attention_mask=masks).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                val_epoch_preds.append(probs)
                val_epoch_targets.append(labels)

        val_epoch_preds = np.concatenate(val_epoch_preds, axis=0)
        val_epoch_targets = np.concatenate(val_epoch_targets, axis=0)

        # Mean Column-wise ROC AUC
        current_auc = roc_auc_score(val_epoch_targets, val_epoch_preds, average='macro')

        if current_auc > best_auc:
            best_auc = current_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load Best Model Weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Step 5: Final Predictions on Validation and Test Sets
    def predict_probs(loader):
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                logits = model(ids, attention_mask=masks).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    val_final_preds = predict_probs(val_loader)
    test_final_preds = predict_probs(test_loader)

    # Format outputs as DataFrames to preserve index alignment and column names
    val_predictions_df = pd.DataFrame(val_final_preds, index=X_val_df.index, columns=y_train_df.columns)
    test_predictions_df = pd.DataFrame(test_final_preds, index=X_test_df.index, columns=y_train_df.columns)

    return val_predictions_df, test_predictions_df


# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "bert_transformer": train_bert_transformer,
}

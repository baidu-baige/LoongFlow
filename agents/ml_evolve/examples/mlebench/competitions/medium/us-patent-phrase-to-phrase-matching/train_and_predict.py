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
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup

BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/aefcc010-8f21-4ecb-b149-7bf99579e6d3/6/executor/output"

# Type hints
DT = pd.DataFrame | pd.Series | np.ndarray
PredictionFunction = Callable[[DT, DT, DT, DT, DT], Tuple[DT, DT]]


# ===== Dataset Class =====

class PatentDataset(Dataset):
    def __init__(self, inputs: pd.DataFrame, labels: pd.Series = None):
        # inputs contains 'input_ids' and 'attention_mask' as lists of ints
        self.input_ids = inputs['input_ids'].values
        self.attention_mask = inputs['attention_mask'].values
        self.labels = labels.values if labels is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        inputs = {
            'input_ids': torch.tensor(self.input_ids[item], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[item], dtype=torch.long),
        }
        if self.labels is not None:
            return inputs, torch.tensor(self.labels[item], dtype=torch.float)
        return inputs


# ===== Model Class =====

class PatentModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # Load pre-trained backbone
        self.model = AutoModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean Pooling: average the hidden states of non-padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        logits = self.fc(mean_embeddings)
        return logits


# ===== Training Functions =====

def train_deberta_v3_large(
    X_train: DT,
    y_train: DT,
    X_val: DT,
    y_val: DT,
    X_test: DT
) -> Tuple[DT, DT]:
    """
    Trains a DeBERTa-v3-large model using Mean Pooling, BCE Loss, and Mixed Precision.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "microsoft/deberta-v3-large"

    # Configuration
    batch_size = 12  # Adjusted for A10 VRAM (23GB) with large model
    epochs = 5
    encoder_lr = 1e-5
    decoder_lr = 5e-5
    weight_decay = 0.01
    eps = 1e-6
    warmup_ratio = 0.1

    # Datasets and Loaders
    train_dataset = PatentDataset(X_train, y_train)
    valid_dataset = PatentDataset(X_val, y_val)
    test_dataset = PatentDataset(X_test, None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model Initialization
    model = PatentModel(model_name).to(device)

    # Differential Learning Rates and Weight Decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.fc.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': decoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.fc.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=encoder_lr, eps=eps)

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()  # Automatic Mixed Precision

    best_score = -1.0
    best_val_preds = None
    best_model_state = None

    # Training Loop
    for epoch in range(epochs):
        model.train()
        for step, (inputs, labels) in enumerate(train_loader):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # AMP
                logits = model(**inputs)
                loss = criterion(logits.view(-1, 1), labels.view(-1, 1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                logits = model(**inputs)
                preds = torch.sigmoid(logits).view(-1).cpu().numpy()
                val_preds.append(preds)
                val_labels.append(labels.numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        # Calculate Pearson Correlation
        score, _ = pearsonr(val_labels, val_preds)

        if score > best_score:
            best_score = score
            best_val_preds = val_preds
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Inference on Test Set using Best Model state
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            logits = model(**inputs)
            preds = torch.sigmoid(logits).view(-1).cpu().numpy()
            test_preds.append(preds)

    test_preds = np.concatenate(test_preds)

    # Cleanup to manage A10 memory
    del model, optimizer, scheduler, train_loader, valid_loader, test_loader, best_model_state, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_preds, test_preds


# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "deberta_v3_large": train_deberta_v3_large,
}

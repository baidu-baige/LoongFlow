import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
import lightgbm as lgb
from scipy.sparse import vstack
from typing import Tuple, Any, Dict, Callable
import socket

# Task-adaptive type definitions
X = pd.DataFrame
y = np.ndarray
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Dataset & Utilities =====

class SpookyDataset(Dataset):
    """General Dataset for Transformer models."""
    def __init__(self, X_df: pd.DataFrame, model_key: str, y_array: np.ndarray = None):
        self.input_ids = np.stack(X_df[f'{model_key}_input_ids'].values)
        self.attention_mask = np.stack(X_df[f'{model_key}_attention_mask'].values)
        self.labels = y_array
        self.indices = np.arange(len(X_df))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'index': torch.tensor(self.indices[idx], dtype=torch.long)
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def find_free_port():
    """Finds a free port for DDP process synchronization."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def calibrate_predictions(val_logits: np.ndarray, y_val: np.ndarray, test_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Applies Platt Scaling using a Logistic Regression calibrator on logits."""
    # We use a high C to approximate unregularized Logistic Regression (Platt Scaling)
    calibrator = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=42)
    calibrator.fit(val_logits, y_val)
    val_probs = calibrator.predict_proba(val_logits)
    test_probs = calibrator.predict_proba(test_logits)
    return val_probs, test_probs

# ===== Transformer DDP Worker =====

def train_transformer_worker(rank, world_size, port, model_name, model_key, X_train, y_train, X_val, y_val, X_test, return_dict):
    """Distributed worker for fine-tuning Transformers (DeBERTa/RoBERTa)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    train_ds = SpookyDataset(X_train, model_key, y_train)
    val_ds = SpookyDataset(X_val, model_key, y_val)
    test_ds = SpookyDataset(X_test, model_key)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

    # Batch size selected to fit A10 (24GB) memory for Large models
    train_loader = DataLoader(train_ds, batch_size=4, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, sampler=val_sampler, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, sampler=test_sampler, num_workers=2, pin_memory=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    model.gradient_checkpointing_enable()
    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    epochs = 4
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * num_training_steps), num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    model.eval()
    def get_logits(loader):
        all_logits, all_indices = [], []
        with torch.no_grad():
            for batch in loader:
                with torch.cuda.amp.autocast():
                    outputs = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
                all_logits.append(outputs.logits.cpu().numpy())
                all_indices.append(batch['index'].numpy())
        if not all_logits: return np.array([]), np.array([])
        return np.concatenate(all_logits), np.concatenate(all_indices)

    v_logits, v_idx = get_logits(val_loader)
    t_logits, t_idx = get_logits(test_loader)
    return_dict[rank] = {'v_logits': v_logits, 'v_idx': v_idx, 't_logits': t_logits, 't_idx': t_idx}
    dist.destroy_process_group()

def run_transformer_training(model_name: str, model_key: str, X_train, y_train, X_val, y_val, X_test) -> Tuple[np.ndarray, np.ndarray]:
    """Orchestrates DDP training and aggregates results across GPUs."""
    world_size = torch.cuda.device_count()
    port = find_free_port()
    manager = mp.Manager()
    return_dict = manager.dict()
    
    mp.spawn(train_transformer_worker, args=(world_size, port, model_name, model_key, X_train, y_train, X_val, y_val, X_test, return_dict), nprocs=world_size, join=True)

    def aggregate(key_logits, key_idx, length):
        l_list = [return_dict[r][key_logits] for r in range(world_size)]
        i_list = [return_dict[r][key_idx] for r in range(world_size)]
        res = pd.DataFrame(np.concatenate(l_list))
        res['idx'] = np.concatenate(i_list)
        return res.drop_duplicates('idx').sort_values('idx').head(length).drop(columns=['idx']).values

    val_logits = aggregate('v_logits', 'v_idx', len(X_val))
    test_logits = aggregate('t_logits', 't_idx', len(X_test))
    
    # Apply Calibration
    print(f"Applying Platt Scaling calibration for {model_name}...")
    return calibrate_predictions(val_logits, y_val, test_logits)

# ===== Training Functions =====

def train_deberta_v3_large(X_train: X, y_train: y, X_val: X, y_val: y, X_test: X) -> Tuple[Predictions, Predictions]:
    print("Fine-tuning DeBERTa-v3-large with DDP and Calibration...")
    return run_transformer_training("microsoft/deberta-v3-large", "deb", X_train, y_train, X_val, y_val, X_test)

def train_roberta_large(X_train: X, y_train: y, X_val: X, y_val: y, X_test: X) -> Tuple[Predictions, Predictions]:
    print("Fine-tuning RoBERTa-large with DDP and Calibration...")
    return run_transformer_training("roberta-large", "rob", X_train, y_train, X_val, y_val, X_test)

def train_logistic_regression(X_train: X, y_train: y, X_val: X, y_val: y, X_test: X) -> Tuple[Predictions, Predictions]:
    print("Training Logistic Regression (C=1.0) on TF-IDF...")
    X_tr = vstack(X_train['tfidf_features'].tolist())
    X_va = vstack(X_val['tfidf_features'].tolist())
    X_te = vstack(X_test['tfidf_features'].tolist())
    model = LogisticRegression(C=1.0, multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_train)
    return model.predict_proba(X_va), model.predict_proba(X_te)

def train_multinomial_nb(X_train: X, y_train: y, X_val: X, y_val: y, X_test: X) -> Tuple[Predictions, Predictions]:
    print("Training Multinomial Naive Bayes (alpha=0.01) on TF-IDF...")
    X_tr = vstack(X_train['tfidf_features'].tolist())
    X_va = vstack(X_val['tfidf_features'].tolist())
    X_te = vstack(X_test['tfidf_features'].tolist())
    model = MultinomialNB(alpha=0.01)
    model.fit(X_tr, y_train)
    return model.predict_proba(X_va), model.predict_proba(X_te)

def train_lightgbm_meta(X_train: X, y_train: y, X_val: X, y_val: y, X_test: X) -> Tuple[Predictions, Predictions]:
    print("Training LightGBM (63 leaves, depth 7) on Meta Features with Calibration...")
    X_tr = np.vstack(X_train['dense_meta'].values)
    X_va = np.vstack(X_val['dense_meta'].values)
    X_te = np.vstack(X_test['dense_meta'].values)

    model = LGBMClassifier(
        objective='multiclass', num_leaves=63, max_depth=7, learning_rate=0.03,
        n_estimators=2000, random_state=42, n_jobs=-1, verbosity=-1
    )
    model.fit(X_tr, y_train, eval_set=[(X_va, y_val)], 
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    
    # LightGBM raw scores (logits) for calibration
    val_logits = model.predict(X_va, raw_score=True)
    test_logits = model.predict(X_te, raw_score=True)
    
    print("Applying Platt Scaling calibration for LightGBM...")
    return calibrate_predictions(val_logits, y_val, test_logits)

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "deberta_v3_large": train_deberta_v3_large,
    "roberta_large": train_roberta_large,
    "logistic_regression": train_logistic_regression,
    "multinomial_nb": train_multinomial_nb,
    "lightgbm_meta": train_lightgbm_meta,
}
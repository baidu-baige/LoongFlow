import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from typing import Tuple, Any, Dict, Callable

# Use the same base paths as defined in the pipeline
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-03/evolux/output/mlebench/jigsaw-toxic-comment-classification-challenge/prepared/public"
OUTPUT_DATA_PATH = "output/4d08636e-bf37-40e0-b9d7-8ffb77d57ea2/1/executor/output"

# Task-adaptive type definitions
X = Any           # ModelInputs (dict-like containing 'input_ids' and 'attention_mask')
y = np.ndarray    # Multi-label binary targets
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

class ToxicDataset(Dataset):
    def __init__(self, inputs: Dict[str, np.ndarray], labels: np.ndarray = None):
        self.input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

def ddp_setup(rank: int, world_size: int):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_worker(
    rank: int, 
    world_size: int, 
    X_train: X, 
    y_train: y, 
    X_val: X, 
    y_val: y, 
    X_test: X, 
    shared_results: Dict
):
    """Worker function for DDP training."""
    ddp_setup(rank, world_size)
    
    # Hyperparameters
    model_name = "microsoft/deberta-v3-base"
    batch_size = 16 # per GPU, total 32
    epochs = 3
    lr = 2e-5
    weight_decay = 0.01

    # Data Preparation
    train_dataset = ToxicDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=DistributedSampler(train_dataset),
        num_workers=4,
        pin_memory=True
    )

    # Model Initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=6
    ).to(rank)
    model = DDP(model, device_ids=[rank])

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    # Loss function (BCEWithLogitsLoss is standard for multi-label)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    print(f"[Rank {rank}] Starting training...")
    model.train()
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    # Inference on Val and Test (Rank 0 handles this for simplicity and consistency)
    if rank == 0:
        print("[Rank 0] Starting inference...")
        model.eval()
        
        def predict(inputs_data):
            dataset = ToxicDataset(inputs_data)
            loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4)
            preds_list = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch['input_ids'].to(rank)
                    attention_mask = batch['attention_mask'].to(rank)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds_list.append(probs)
            return np.concatenate(preds_list, axis=0)

        val_preds = predict(X_val)
        test_preds = predict(X_test)
        
        shared_results['val_preds'] = val_preds
        shared_results['test_preds'] = test_preds

    destroy_process_group()

def train_deberta_v3(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains DeBERTa-v3 using Distributed Data Parallel on available GPUs.
    """
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU available for training.")
    
    print(f"Initializing DDP with {world_size} GPUs.")
    
    # Use a manager to retrieve results from the worker processes
    manager = mp.Manager()
    shared_results = manager.dict()

    # Spawn processes
    mp.spawn(
        train_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, shared_results),
        nprocs=world_size,
        join=True
    )

    if 'val_preds' not in shared_results or 'test_preds' not in shared_results:
        raise RuntimeError("Training failed to produce predictions.")

    val_preds = shared_results['val_preds']
    test_preds = shared_results['test_preds']

    # Final sanity checks
    if np.isnan(val_preds).any() or np.isnan(test_preds).any():
        raise ValueError("Predictions contain NaN values.")

    print("DeBERTa-v3 training and inference completed successfully.")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "deberta_v3": train_deberta_v3,
}
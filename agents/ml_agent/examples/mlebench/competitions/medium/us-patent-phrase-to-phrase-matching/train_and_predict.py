import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoConfig, get_cosine_schedule_with_warmup
import pandas as pd
import numpy as np
import os
import gc
from typing import Tuple, Any, Dict, Callable

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/us-patent-phrase-to-phrase-matching/prepared/public"
OUTPUT_DATA_PATH = "output/02d42284-9bf3-4f97-ab6c-7ea839095b54/3/executor/output"

# ===== PyTorch Dataset =====

class PatentDataset(Dataset):
    """
    Dataset class for patent phrase matching. Handles tokenized inputs and scores.
    """
    def __init__(self, X_df: pd.DataFrame, y_ser: pd.Series = None):
        # Convert Series/DataFrame columns to lists for efficient indexing
        self.input_ids = X_df['input_ids'].tolist()
        self.attention_mask = X_df['attention_mask'].tolist()
        self.token_type_ids = X_df['token_type_ids'].tolist() if 'token_type_ids' in X_df.columns else None
        self.targets = y_ser.values if y_ser is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.token_type_ids is not None:
            item['token_type_ids'] = torch.tensor(self.token_type_ids[idx], dtype=torch.long)
        
        if self.targets is not None:
            item['target'] = torch.tensor(self.targets[idx], dtype=torch.float)
        
        return item

# ===== Model Definition =====

class PatentModel(nn.Module):
    """
    DeBERTa-v3-large model with a regression head and Multi-sample Dropout.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        # Multi-sample dropout: 5 samples with increasing dropout rates
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Use [CLS] token representation
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        
        # Multi-sample dropout averaging
        logits = torch.stack([
            self.fc(dropout(last_hidden_state)) for dropout in self.dropouts
        ], dim=0).mean(dim=0)
        
        return logits.view(-1)

# ===== DDP Worker Function =====

def ddp_worker(rank: int, world_size: int, X_train, y_train, X_val, y_val, X_test, queue):
    """
    Worker function for Distributed Data Parallel training.
    """
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Technical Specification Parameters
    model_name = "microsoft/deberta-v3-large"
    batch_size = 8 # per GPU
    epochs = 5
    lr_backbone = 2e-5
    lr_head = 5e-5
    
    # Datasets
    train_ds = PatentDataset(X_train, y_train)
    val_ds = PatentDataset(X_val, y_val)
    test_ds = PatentDataset(X_test)
    
    # Samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    
    # DataLoaders
    # Validation and test are small enough to process on each rank or rank 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    # Model and DDP setup
    model = PatentModel(model_name).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # BCEWithLogitsLoss for regression in range [0, 1]
    criterion = nn.BCEWithLogitsLoss()
    
    # Differential learning rates
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.module.model.named_parameters()], 'lr': lr_backbone},
        {'params': [p for n, p in model.module.named_parameters() if "model" not in n], 'lr': lr_head},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    num_train_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

    # Training Loop
    print(f"[Rank {rank}] Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(rank)
            mask = batch['attention_mask'].to(rank)
            ttid = batch.get('token_type_ids', None)
            if ttid is not None: ttid = ttid.to(rank)
            targets = batch['target'].to(rank)
            
            outputs = model(ids, mask, ttid)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Inference (Rank 0 handles final predictions)
    model.eval()
    val_preds = []
    test_preds = []
    
    if rank == 0:
        print(f"[Rank 0] Training finished. Starting inference...")
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(rank)
                mask = batch['attention_mask'].to(rank)
                ttid = batch.get('token_type_ids', None)
                if ttid is not None: ttid = ttid.to(rank)
                outputs = model(ids, mask, ttid)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
        val_preds = np.concatenate(val_preds)

        with torch.no_grad():
            for batch in test_loader:
                ids = batch['input_ids'].to(rank)
                mask = batch['attention_mask'].to(rank)
                ttid = batch.get('token_type_ids', None)
                if ttid is not None: ttid = ttid.to(rank)
                outputs = model(ids, mask, ttid)
                test_preds.append(torch.sigmoid(outputs).cpu().numpy())
        test_preds = np.concatenate(test_preds)
        
        # Push results to queue
        queue.put((val_preds, test_preds))

    dist.destroy_process_group()

# ===== Training Functions =====

def train_deberta_v3_large_v2(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains DeBERTa-v3-large with 5 epochs and Multi-sample Dropout using multi-GPU DDP.
    """
    print(f"Execution: train_deberta_v3_large_v2 (Stage 4)")
    
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU available for training. This component requires GPU acceleration.")
    
    print(f"Initializing DDP on {world_size} GPUs.")

    # Using 'spawn' context for multi-processing
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    
    # Spawn processes across available GPUs
    mp.spawn(
        ddp_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, queue),
        nprocs=world_size,
        join=True
    )
    
    # Retrieve results from Rank 0
    if not queue.empty():
        val_preds, test_preds = queue.get()
    else:
        raise RuntimeError("Training failed: No predictions returned from DDP workers.")

    # Verify output quality
    if np.isnan(val_preds).any() or np.isinf(val_preds).any():
        raise ValueError("Validation predictions contain NaN or Infinity.")
    if np.isnan(test_preds).any() or np.isinf(test_preds).any():
        raise ValueError("Test predictions contain NaN or Infinity.")

    # Resource cleanup
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Training complete. Val preds: {len(val_preds)}, Test preds: {len(test_preds)}")
    return val_preds, test_preds

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "deberta_v3_large_v2": train_deberta_v3_large_v2,
}
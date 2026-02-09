import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup, AutoTokenizer
import pandas as pd
import numpy as np
import os
import random
from typing import Tuple, Any, Dict, Callable

# BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-01/evolux/output/mlebench/lmsys-chatbot-arena/prepared/public"
# OUTPUT_DATA_PATH = "output/e051fb32-5ec2-4424-8b42-87dd7b28dacc/1/executor/output"

# Task-adaptive type definitions for LMSYS Chatbot Arena
X = pd.DataFrame
y = np.ndarray
Predictions = np.ndarray
ModelFn = Callable[[X, y, X, y, X], Tuple[Predictions, Predictions]]

class ChatbotDataset(Dataset):
    """
    Dataset for Chatbot Arena preference prediction.
    Supports on-the-fly augmentation by swapping response A and B components.
    """
    def __init__(self, input_ids, attention_mask, labels=None, augment=False, sep_token_id=2):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.augment = augment
        self.sep_token_id = sep_token_id

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = list(self.input_ids[idx])
        mask = list(self.attention_mask[idx])
        
        label = -1
        if self.labels is not None:
            label = self.labels[idx]
            # 50% Position Swap: Swap response_a and response_b tokens and flip labels 0/1
            if self.augment and random.random() < 0.5:
                # Sequence format created in preprocess: [CLS] prompt [SEP] response_a [SEP] response_b [SEP]
                sep_indices = [i for i, tid in enumerate(ids) if tid == self.sep_token_id]
                if len(sep_indices) >= 3:
                    s1, s2, s3 = sep_indices[0], sep_indices[1], sep_indices[2]
                    # Swap segments: response_a + [SEP] and response_b + [SEP]
                    # Indices: ids[:s1+1] is prompt part, ids[s1+1:s2+1] is ra, ids[s2+1:s3+1] is rb
                    new_ids = ids[:s1+1] + ids[s2+1:s3+1] + ids[s1+1:s2+1] + ids[s3+1:]
                    ids = new_ids
                    # Flip labels: 0 (winner_a) <-> 1 (winner_b), 2 (tie) stays 2
                    if label == 0: label = 1
                    elif label == 1: label = 0
        
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def worker(rank, world_size, X_train, y_train, X_val, X_test, sep_token_id, return_dict):
    """
    Distributed training worker process utilizing NCCL backend.
    """
    try:
        # 1. Initialize Process Group for DDP
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        
        # 2. Reproducibility within ranks
        seed = 42 + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 3. Model Architecture and Configuration
        model_name = "microsoft/deberta-v3-large"
        config = AutoConfig.from_pretrained(model_name, num_labels=3)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        
        # Optimization: Gradient Checkpointing is mandatory for sequence length 1536
        model.gradient_checkpointing_enable()
        model.to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])
        
        # 4. Data Preparation
        train_ds = ChatbotDataset(
            X_train['input_ids'].values, 
            X_train['attention_mask'].values, 
            y_train, 
            augment=True, 
            sep_token_id=sep_token_id
        )
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        # Global batch_size 16 -> 8 per GPU
        train_loader = DataLoader(train_ds, batch_size=8, sampler=train_sampler, num_workers=2, pin_memory=True)
        
        # 5. Optimization Strategy
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        num_epochs = 2
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps), 
            num_training_steps=total_steps
        )
        
        # 6. Training Loop with FP16 Mixed Precision
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(num_epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            for batch in train_loader:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch['input_ids'].to(rank),
                        attention_mask=batch['attention_mask'].to(rank),
                        labels=batch['labels'].to(rank)
                    )
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        
        # 7. Final Inference Phase
        # Only Rank 0 performs full set inference to simplify result collection
        if rank == 0:
            model.eval()
            def predict_full_set(X_df):
                if len(X_df) == 0:
                    return np.empty((0, 3))
                ds = ChatbotDataset(X_df['input_ids'].values, X_df['attention_mask'].values, sep_token_id=sep_token_id)
                loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
                probs_list = []
                with torch.no_grad():
                    for batch in loader:
                        with torch.cuda.amp.autocast():
                            logits = model(
                                input_ids=batch['input_ids'].to(rank),
                                attention_mask=batch['attention_mask'].to(rank)
                            ).logits
                            probs = torch.softmax(logits, dim=1)
                            probs_list.append(probs.cpu().numpy())
                return np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, 3))

            print("Rank 0: Generating validation and test predictions...")
            return_dict['val_preds'] = predict_full_set(X_val)
            return_dict['test_preds'] = predict_full_set(X_test)

    finally:
        dist.destroy_process_group()

def train_deberta_v3_large(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains DeBERTa-v3-large using 2-GPU DistributedDataParallel and generates predictions.
    """
    print("Initializing DeBERTa-v3-large Training Engine...")
    
    # Identify SEP token ID for augmentation logic
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sep_token_id = tokenizer.sep_token_id
    
    # Resource utilization check
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Warning: Only {world_size} GPU(s) found. Pipeline requires 2 for optimal performance.")
        if world_size == 0:
            raise RuntimeError("No GPUs detected.")
    
    # Use Manager for communicating predictions from child processes
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Spawn distributed worker processes
    mp.spawn(
        worker,
        args=(world_size, X_train, y_train, X_val, X_test, sep_token_id, return_dict),
        nprocs=world_size,
        join=True
    )
    
    # Verify results
    if 'val_preds' not in return_dict or 'test_preds' not in return_dict:
        raise RuntimeError("Distributed training failed to return validation or test predictions.")
        
    val_preds = return_dict['val_preds']
    test_preds = return_dict['test_preds']
    
    print(f"Training Complete. Validation Shape: {val_preds.shape}, Test Shape: {test_preds.shape}")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "deberta_v3_large": train_deberta_v3_large,
}
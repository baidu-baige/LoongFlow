import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from typing import Tuple, Any, Dict, Callable

# Target paths
BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/h20-12/evolux/output/mlebench/chaii-hindi-and-tamil-question-answering/prepared/public"
OUTPUT_DATA_PATH = "output/af0f7d71-a062-46e3-8926-51aedd28d3b4/3/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame      # Preprocessed chunks
y = pd.DataFrame      # Preprocessed targets
Predictions = np.ndarray # Logits (N_chunks, seq_len, 2)

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== Model Definition =====

class IndianQAModel(nn.Module):
    """
    Extractive QA model with Multi-sample Dropout for regularization.
    Compatible with XLM-RoBERTa and MuRIL backbones.
    """
    def __init__(self, model_name: str, num_samples: int = 5):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.num_samples = num_samples
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        
        if self.training:
            # Multi-sample dropout: averaging logits from multiple dropout passes
            logits = 0
            for _ in range(self.num_samples):
                logits += self.classifier(self.dropout(sequence_output))
            logits = logits / self.num_samples
        else:
            logits = self.classifier(sequence_output)
            
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        loss = None
        if start_positions is not None and end_positions is not None:
            # Standard CrossEntropy for start and end token positions
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            
        return loss, start_logits, end_logits

# ===== Dataset Helper =====

class QADataset(Dataset):
    def __init__(self, X_df: pd.DataFrame, y_df: pd.DataFrame, model_prefix: str):
        """
        model_prefix: 'xlmr' or 'muril' to select the correct preprocessed columns.
        """
        self.input_ids = np.stack(X_df[f"{model_prefix}_input_ids"].values)
        self.attention_mask = np.stack(X_df[f"{model_prefix}_attention_mask"].values)
        self.has_targets = y_df is not None and not y_df.empty
        if self.has_targets:
            self.start_positions = y_df[f"{model_prefix}_start_positions"].values
            self.end_positions = y_df[f"{model_prefix}_end_positions"].values
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long)
        }
        if self.has_targets:
            item["start_positions"] = torch.tensor(self.start_positions[idx], dtype=torch.long)
            item["end_positions"] = torch.tensor(self.end_positions[idx], dtype=torch.long)
        return item

# ===== Optimizer Setup =====

def get_optimizer_params(model, lr, weight_decay, layerwise_lr_decay):
    """
    Implements Layer-wise Learning Rate Decay (LLRD).
    Higher layers (closer to the output) have higher learning rates.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Start with the classifier (top-most layer)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    
    # Access backbone layers
    backbone = getattr(model, "backbone", None)
    if not backbone and hasattr(model, "module"):
        backbone = model.module.backbone
        
    if backbone and hasattr(backbone, "encoder"):
        # Layers: [embeddings, layer 0, ..., layer N]
        layers = [backbone.embeddings] + list(backbone.encoder.layer)
        layers.reverse() # [layer N, ..., layer 0, embeddings]
        
        current_lr = lr
        for layer in layers:
            current_lr *= layerwise_lr_decay
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": current_lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": current_lr,
                },
            ]
        return optimizer_grouped_parameters
    else:
        return model.parameters()

# ===== Training Worker for DDP =====

def train_worker(rank, world_size, X_train, y_train, X_val, y_val, X_test, model_config, shared_results):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Unpack config
    model_name = model_config["name"]
    model_prefix = model_config["prefix"]
    lr = model_config["lr"]
    epochs = 3
    batch_size = 16 # per GPU (Total 32)
    weight_decay = 0.01
    llrd_factor = 0.9
    
    # Datasets
    train_ds = QADataset(X_train, y_train, model_prefix)
    val_ds = QADataset(X_val, y_val, model_prefix)
    test_ds = QADataset(X_test, None, model_prefix)
    
    # DDP Sampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    
    # Model Setup
    model = IndianQAModel(model_name, num_samples=5).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Optimizer & Scheduler
    opt_params = get_optimizer_params(model.module, lr, weight_decay, llrd_factor)
    optimizer = optim.AdamW(opt_params)
    num_train_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_train_steps), 
        num_training_steps=num_train_steps
    )
    
    # Training Loop
    if rank == 0: print(f"Starting training for {model_name}...")
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            loss, _, _ = model(input_ids, attention_mask, start_positions, end_positions)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if rank == 0: print(f"Epoch {epoch+1}/{epochs} complete.")
            
    # Inference on Rank 0
    if rank == 0:
        model.eval()
        # Use standard loaders to maintain sample order for validation/test sets
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
        
        def run_inference(loader):
            preds = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    _, start_logits, end_logits = model.module(input_ids, attention_mask)
                    # Shape: (batch, seq_len, 2)
                    logits = torch.stack([start_logits, end_logits], dim=-1)
                    preds.append(logits.cpu().numpy())
            return np.concatenate(preds, axis=0)

        shared_results['val'] = run_inference(val_loader)
        shared_results['test'] = run_inference(test_loader)
        
    dist.barrier()
    dist.destroy_process_group()

# ===== Training Master Functions =====

def _train_model_common(X_train, y_train, X_val, y_val, X_test, model_config):
    """
    Common entry point for training transformer-based QA models using DDP.
    """
    world_size = torch.cuda.device_count()
    if world_size < 2:
        # Fallback for single GPU, but pipeline guarantees 2
        world_size = max(1, world_size)
        
    manager = mp.Manager()
    shared_results = manager.dict()
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(20000, 30000))
    
    mp.spawn(
        train_worker,
        args=(world_size, X_train, y_train, X_val, y_val, X_test, model_config, shared_results),
        nprocs=world_size,
        join=True
    )
    
    val_preds = shared_results.get('val')
    test_preds = shared_results.get('test')
    
    if val_preds is None or test_preds is None:
        raise RuntimeError(f"Training failed to produce results for {model_config['name']}")
        
    return val_preds, test_preds

def train_muril_large_qa(X_train: X, y_train: y, X_val: X, y_val: y, X_test: X) -> Tuple[Predictions, Predictions]:
    """Trains Google MuRIL-Large for Indic QA."""
    config = {
        "name": "google/muril-large-cased",
        "prefix": "muril",
        "lr": 2e-5
    }
    print("Initializing MuRIL-Large training workflow...")
    return _train_model_common(X_train, y_train, X_val, y_val, X_test, config)

def train_xlm_roberta_large_qa(X_train: X, y_train: y, X_val: X, y_val: y, X_test: X) -> Tuple[Predictions, Predictions]:
    """Trains XLM-RoBERTa-Large for Multilingual QA."""
    config = {
        "name": "xlm-roberta-large",
        "prefix": "xlmr",
        "lr": 1.5e-5
    }
    print("Initializing XLM-RoBERTa-Large training workflow...")
    return _train_model_common(X_train, y_train, X_val, y_val, X_test, config)

# ===== Model Registry =====

PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "muril_large_qa": train_muril_large_qa,
    "xlm_roberta_large_qa": train_xlm_roberta_large_qa,
}
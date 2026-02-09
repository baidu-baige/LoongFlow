import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
from typing import Tuple, Dict, Any, Callable, List
import copy

# Task-adaptive type definitions for DeBERTa Extractive QA
X = Dict[str, np.ndarray]           # {'input_ids': np.ndarray, 'attention_mask': np.ndarray}
y = Dict[str, np.ndarray]           # {'start_positions': np.ndarray, 'end_positions': np.ndarray}
Predictions = Dict[str, np.ndarray] # {'start_logits': np.ndarray, 'end_logits': np.ndarray}

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/1-04/evolux/output/mlebench/tweet-sentiment-extraction/prepared/public"
OUTPUT_DATA_PATH = "output/1cc1b6ac-193c-4f3f-a388-380b832f53e8/5/executor/output"

class TweetDataset(Dataset):
    """
    Dataset wrapper for tokenized tweet inputs and span indices.
    """
    def __init__(self, features: X, targets: y = None):
        self.input_ids = features['input_ids']
        self.attention_mask = features['attention_mask']
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'sample_idx': torch.tensor(idx, dtype=torch.long)
        }
        if self.targets is not None:
            item['start_positions'] = torch.tensor(self.targets['start_positions'][idx], dtype=torch.long)
            item['end_positions'] = torch.tensor(self.targets['end_positions'][idx], dtype=torch.long)
        return item

class DebertaV3LargeQA(nn.Module):
    """
    DeBERTa-v3-Large model architecture for Extractive QA with Vectorized Multi-Sample Dropout.
    """
    def __init__(self, model_name='microsoft/deberta-v3-large', num_dropout_samples=5, dropout_rate=0.5):
        super().__init__()
        # Load DeBERTa-v3-large backbone (DebertaV2Model)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_dropout_samples)])
        # Classifier producing 2 logits (start and end) per token. 
        # DeBERTa-v3-large hidden dimension is 1024.
        self.classifier = nn.Linear(1024, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, sequence_length, 1024)
        
        batch_size = sequence_output.size(0)
        seq_len = sequence_output.size(1)
        
        # Vectorized implementation of Multi-Sample Dropout to maintain a clean autograd graph for DDP
        # (num_samples, batch_size, seq_len, 1024)
        x_multi = torch.stack([dropout(sequence_output) for dropout in self.dropouts], dim=0)
        
        # Reshape to (num_samples * batch_size * seq_len, 1024) for efficient linear layer application
        x_flat = x_multi.view(-1, 1024)
        logits_flat = self.classifier(x_flat)
        
        # Reshape back and average across dropout samples: (batch_size, seq_len, 2)
        logits = logits_flat.view(len(self.dropouts), batch_size, seq_len, 2).mean(0)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

def average_checkpoints(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Computes the arithmetic average of provided state dictionaries."""
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] += state_dicts[i][key]
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(state_dicts))
    return avg_state_dict

def _train_worker(rank: int, world_size: int, train_data: Tuple[X, y], val_data: Tuple[X, y], test_data: X, results_queue: mp.Queue):
    """
    Worker process for Distributed Data Parallel (DDP) training of DeBERTa-v3.
    """
    try:
        # Step 0: Distributed Initialization
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12365'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        # Step 1: Data Preparation
        train_ds = TweetDataset(train_data[0], train_data[1])
        val_ds = TweetDataset(val_data[0]) 
        test_ds = TweetDataset(test_data)

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)

        # Batch size 8 per GPU (Total 16)
        train_loader = DataLoader(train_ds, batch_size=8, sampler=train_sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=16, sampler=val_sampler, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=16, sampler=test_sampler, num_workers=2, pin_memory=True)

        # Step 2: Model and Optimization Setup
        model = DebertaV3LargeQA(model_name='microsoft/deberta-v3-large').to(rank)
        
        # TECHNICAL FIX: Enable gradient checkpointing to manage memory on 24GB A10 GPUs.
        model.backbone.gradient_checkpointing_enable()
        
        # TECHNICAL FIX: Use static_graph=True to resolve DDP-checkpointing synchronization issues 
        # ("Expected to mark a variable ready only once").
        model = DDP(model, device_ids=[rank], find_unused_parameters=True, static_graph=True)

        epochs = 3
        learning_rate = 1e-5
        
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        num_train_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_train_steps * 0.1), num_training_steps=num_train_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        saved_state_dicts = []

        # Step 3: Training Loop
        print(f"[Rank {rank}] Training DeBERTa-v3-Large...")
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                ids = batch['input_ids'].to(rank)
                mask = batch['attention_mask'].to(rank)
                s_true = batch['start_positions'].to(rank)
                e_true = batch['end_positions'].to(rank)
                
                s_logits, e_logits = model(ids, mask)
                loss = (criterion(s_logits, s_true) + criterion(e_logits, e_true)) / 2
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            
            if rank == 0:
                print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(train_loader):.4f}")
            
            # Capture state dicts for Epoch 2 and 3 for weight averaging
            if epoch >= 1:
                state_dict = {k: v.cpu() for k, v in model.module.state_dict().items()}
                saved_state_dicts.append(state_dict)

        # Step 4: Weight Averaging (Stochastic Weight Averaging variant)
        if rank == 0:
            print(f"Rank {rank}: Computing averaged weights.")
        avg_sd = average_checkpoints(saved_state_dicts)
        model.module.load_state_dict(avg_sd)

        # Step 5: Distributed Inference
        def run_inference(loader):
            model.eval()
            starts, ends, indices = [], [], []
            with torch.no_grad():
                for batch in loader:
                    ids = batch['input_ids'].to(rank)
                    mask = batch['attention_mask'].to(rank)
                    idx = batch['sample_idx']
                    s_logits, e_logits = model(ids, mask)
                    starts.append(s_logits.cpu().numpy())
                    ends.append(e_logits.cpu().numpy())
                    indices.append(idx.numpy())
            
            if not starts:
                return None, None, None
            return np.concatenate(starts), np.concatenate(ends), np.concatenate(indices)

        val_s, val_e, val_idx = run_inference(val_loader)
        test_s, test_e, test_idx = run_inference(test_loader)

        results_queue.put((rank, val_s, val_e, val_idx, test_s, test_e, test_idx))
        
    except Exception as e:
        print(f"[Rank {rank}] Worker Error: {str(e)}")
        results_queue.put((rank, None, None, None, None, None, None))
        raise e
    finally:
        dist.destroy_process_group()

def train_deberta_v3_large_qa(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Orchestrates DeBERTa-v3-Large training with checkpoint averaging and multi-sample dropout.
    Leverages 2 GPUs via DistributedDataParallel with static_graph optimization.
    """
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Warning: GPU count {world_size} < 2. Adjusting.")
        world_size = max(1, world_size)

    print(f"Initializing distributed training on {world_size} GPUs...")
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    mp.spawn(
        _train_worker,
        args=(world_size, (X_train, y_train), (X_val, y_val), X_test, results_queue),
        nprocs=world_size,
        join=True
    )

    # Reconstruct predictions from worker shards
    val_len = len(X_val['input_ids'])
    test_len = len(X_test['input_ids'])
    seq_len = X_train['input_ids'].shape[1]
    
    final_val_s = np.zeros((val_len, seq_len), dtype=np.float32)
    final_val_e = np.zeros((val_len, seq_len), dtype=np.float32)
    final_test_s = np.zeros((test_len, seq_len), dtype=np.float32)
    final_test_e = np.zeros((test_len, seq_len), dtype=np.float32)

    for _ in range(world_size):
        rank, v_s, v_e, v_idx, t_s, t_e, t_idx = results_queue.get()
        if v_s is None:
            raise RuntimeError(f"Worker at rank {rank} failed during execution.")

        # Re-map predictions using sample indices
        for i, idx in enumerate(v_idx):
            if idx < val_len:
                final_val_s[idx] = v_s[i]
                final_val_e[idx] = v_e[i]
        
        for i, idx in enumerate(t_idx):
            if idx < test_len:
                final_test_s[idx] = t_s[i]
                final_test_e[idx] = t_e[i]

    # Verify output integrity
    if np.isnan(final_val_s).any():
        raise ValueError("Model produced NaN values in validation predictions.")

    val_preds = {'start_logits': final_val_s, 'end_logits': final_val_e}
    test_preds = {'start_logits': final_test_s, 'end_logits': final_test_e}

    print("Training and inference complete.")
    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "deberta_v3_large_qa": train_deberta_v3_large_qa,
}
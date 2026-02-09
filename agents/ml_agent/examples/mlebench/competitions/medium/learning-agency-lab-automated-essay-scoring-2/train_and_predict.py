import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
import lightgbm as lgb
from scipy import sparse
from typing import Tuple, Any, Dict, Callable

# Task-adaptive type definitions
X = sparse.csr_matrix
y = np.ndarray
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== DeBERTa Model Components =====

class EssayDataset(Dataset):
    def __init__(self, ids, mask, targets=None):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        item = {'input_ids': self.ids[idx], 'attention_mask': self.mask[idx]}
        if self.targets is not None:
            item['labels'] = self.targets[idx]
        return item

class DebertaRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        # Truncate to model max length (e.g., 512 for deberta-v3-base)
        max_len = self.config.max_position_embeddings
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Mean Pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooling = sum_embeddings / sum_mask
        
        return self.head(mean_pooling).squeeze(-1)

def get_optimizer_params(model, encoder_lr, head_lr, weight_decay, llrd_factor):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.head.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay, "lr": head_lr,
        },
        {
            "params": [p for n, p in model.head.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": head_lr,
        },
    ]
    layers = [model.deberta.embeddings] + list(model.deberta.encoder.layer)
    layers.reverse()
    lr = encoder_lr
    for layer in layers:
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay, "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0, "lr": lr,
            },
        ]
        lr *= llrd_factor
    return optimizer_grouped_parameters

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_worker(rank, world_size, train_ids, train_mask, train_y, val_ids, val_mask, test_ids, test_mask, return_dict):
    ddp_setup(rank, world_size)
    
    model_name = "microsoft/deberta-v3-base"
    model = DebertaRegressor(model_name).to(rank)
    model = DDP(model, device_ids=[rank])
    
    train_dataset = EssayDataset(train_ids, train_mask, train_y)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, num_workers=0, pin_memory=True)
    
    optimizer_params = get_optimizer_params(model.module, encoder_lr=2e-5, head_lr=2e-5, weight_decay=0.01, llrd_factor=0.9)
    optimizer = torch.optim.AdamW(optimizer_params)
    
    epochs = 3
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    print(f"[Rank {rank}] Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(batch['input_ids'].to(rank), batch['attention_mask'].to(rank))
                loss = criterion(outputs, batch['labels'].to(rank))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    # Inference on Rank 0
    if rank == 0:
        model.eval()
        def predict_dataloader(ids, mask):
            ds = EssayDataset(ids, mask)
            dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            all_preds = []
            with torch.no_grad():
                for b in dl:
                    with torch.cuda.amp.autocast():
                        preds = model(b['input_ids'].to(rank), b['attention_mask'].to(rank))
                    all_preds.append(preds.cpu().numpy())
            return np.concatenate(all_preds)

        print(f"[Rank 0] Generating DeBERTa predictions...")
        return_dict['deberta_val'] = predict_dataloader(val_ids, val_mask)
        return_dict['deberta_test'] = predict_dataloader(test_ids, test_mask)

    dist.destroy_process_group()

# ===== Main Training Function =====

def train_deberta_lgbm(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains DeBERTa-v3 and LightGBM regressors and returns averaged predictions.
    """
    print("Stage 4: Training DeBERTa-v3 and LightGBM...")
    
    # 1. Prepare Data
    # Index mapping from preprocess.py:
    # GBDT [0:20005], IDs [20005:21029], Mask [21029:22053]
    X_train_gbdt = X_train[:, :20005]
    X_val_gbdt = X_val[:, :20005]
    X_test_gbdt = X_test[:, :20005]
    
    ids_train = X_train[:, 20005:21029].toarray().astype(np.int64)
    mask_train = X_train[:, 21029:22053].toarray().astype(np.int64)
    ids_val = X_val[:, 20005:21029].toarray().astype(np.int64)
    mask_val = X_val[:, 21029:22053].toarray().astype(np.int64)
    ids_test = X_test[:, 20005:21029].toarray().astype(np.int64)
    mask_test = X_test[:, 21029:22053].toarray().astype(np.int64)
    
    y_train_f = y_train.astype(np.float32)

    # 2. Train DeBERTa via DDP (2 GPUs)
    manager = mp.Manager()
    return_dict = manager.dict()
    world_size = 2
    
    print("Spawning DeBERTa DDP workers...")
    mp.spawn(ddp_worker, 
             args=(world_size, ids_train, mask_train, y_train_f, ids_val, mask_val, ids_test, mask_test, return_dict), 
             nprocs=world_size, 
             join=True)
    
    deberta_val = return_dict['deberta_val']
    deberta_test = return_dict['deberta_test']

    # 3. Train LightGBM (GPU)
    print("Training LightGBM on GPU...")
    lgbm_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1500,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'colsample_bytree': 0.5,
        'device': 'cuda',
        'random_state': 42,
        'verbosity': -1
    }
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train_gbdt, y_train_f)
    
    lgbm_val = lgbm_model.predict(X_val_gbdt)
    lgbm_test = lgbm_model.predict(X_test_gbdt)

    # 4. Ensemble and Return
    val_preds = (0.5 * deberta_val + 0.5 * lgbm_val)
    test_preds = (0.5 * deberta_test + 0.5 * lgbm_test)
    
    # Sanity check for NaNs
    val_preds = np.nan_to_num(val_preds, nan=np.mean(y_train))
    test_preds = np.nan_to_num(test_preds, nan=np.mean(y_train))

    print(f"Training Complete. Val Preds Mean: {val_preds.mean():.4f}, Test Preds Mean: {test_preds.mean():.4f}")
    
    # Cleanup
    del lgbm_model
    gc.collect()
    torch.cuda.empty_cache()

    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "deberta_lgbm_regressor": train_deberta_lgbm,
}
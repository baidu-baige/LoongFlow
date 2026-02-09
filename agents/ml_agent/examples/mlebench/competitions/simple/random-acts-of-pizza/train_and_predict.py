from typing import Tuple, Any, Dict, Callable
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/random-acts-of-pizza/prepared/public"
OUTPUT_DATA_PATH = "output/f2dbb22d-a0cb-4add-aa87-f2c6b1a4b76f/77/executor/output"

# Task-adaptive type definitions
X = pd.DataFrame
y = pd.Series
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# ===== DeBERTa Components =====

class PizzaDataset(Dataset):
    def __init__(self, texts, meta, labels=None, tokenizer=None, max_len=256):
        self.texts = texts.values
        self.meta = meta.values.astype(np.float32)
        self.labels = labels.values.astype(np.float32) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'meta': torch.tensor(self.meta[idx])
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

class DebertaHybridModel(nn.Module):
    def __init__(self, model_name, meta_dim):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.deberta.config.hidden_size + meta_dim, 1)

    def forward(self, input_ids, attention_mask, meta):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_output, meta], dim=1)
        logits = self.classifier(self.dropout(combined))
        return logits

# ===== Training Function =====

def train_hybrid_pseudo_label_ensemble(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    """
    Trains a heterogeneous ensemble of a Hybrid DeBERTa specialist and pseudo-labeled tabular models.
    Returns the averaged predictions of the 5 specialists.
    """
    print("Initializing Hybrid Pseudo-Labeling Strategy...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data
    # Identify metadata and text
    meta_cols = ['log_age', 'log_karma', 'is_giver_known']
    available_meta = [c for c in meta_cols if c in X_train.columns]
    if not available_meta:
        available_meta = [c for c in X_train.columns if c != 'raw_text' and np.issubdtype(X_train[c].dtype, np.number)][:10]
    
    X_train_tab = X_train.drop(columns=['raw_text'])
    X_val_tab = X_val.drop(columns=['raw_text'])
    X_test_tab = X_test.drop(columns=['raw_text'])
    
    # 2. Hybrid DeBERTa Specialist
    print("Training Hybrid DeBERTa Specialist...")
    model_name = 'microsoft/deberta-v3-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_ds = PizzaDataset(X_train['raw_text'], X_train[available_meta], y_train, tokenizer)
    val_ds = PizzaDataset(X_val['raw_text'], X_val[available_meta], y_val, tokenizer)
    test_ds = PizzaDataset(X_test['raw_text'], X_test[available_meta], None, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)
    
    model = DebertaHybridModel(model_name, len(available_meta)).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['meta'].to(device))
            loss = criterion(logits.squeeze(), batch['labels'].to(device))
            loss.backward()
            optimizer.step()
        print(f"DeBERTa Epoch {epoch+1} Complete")
        
    def get_deb_preds(loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['meta'].to(device))
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(preds).flatten()

    val_preds_deb = get_deb_preds(val_loader)
    test_preds_deb = get_deb_preds(test_loader)

    # 3. Seed Tabular Training & Pseudo-Labeling
    print("Seed Training and Pseudo-Labeling...")
    # Using CPU for LGBM due to environment build constraints; GPU for CatBoost/XGBoost
    models_cfg = {
        "lgbm": LGBMClassifier(n_estimators=1000, learning_rate=0.01, is_unbalance=True, n_jobs=-1, random_state=42, verbosity=-1),
        "catboost": CatBoostClassifier(iterations=1000, learning_rate=0.01, task_type='GPU', devices='0:1:2:3', auto_class_weights='Balanced', random_seed=42, verbose=False),
        "xgboost": XGBClassifier(n_estimators=1000, learning_rate=0.01, device='cuda', scale_pos_weight=3, random_state=42),
        "lr": LogisticRegression(class_weight='balanced', C=0.1, max_iter=1000, random_state=42)
    }
    
    seed_test_preds = []
    for name, m in models_cfg.items():
        m.fit(X_train_tab, y_train)
        seed_test_preds.append(m.predict_proba(X_test_tab)[:, 1])
    
    avg_seed_test = np.mean(seed_test_preds, axis=0)
    high_conf_pos = avg_seed_test > 0.85
    high_conf_neg = avg_seed_test < 0.15
    
    X_test_pseudo = X_test_tab[high_conf_pos | high_conf_neg].copy()
    y_test_pseudo = pd.Series(np.where(high_conf_pos[high_conf_pos | high_conf_neg], 1, 0), index=X_test_pseudo.index)
    
    X_train_aug = pd.concat([X_train_tab, X_test_pseudo])
    y_train_aug = pd.concat([y_train, y_test_pseudo])
    
    # 4. Final Retraining and Aggregation
    print(f"Retraining 4 tabular specialists on augmented set (+{len(X_test_pseudo)} samples)...")
    all_val_preds = [val_preds_deb]
    all_test_preds = [test_preds_deb]
    
    for name, m in models_cfg.items():
        m.fit(X_train_aug, y_train_aug)
        all_val_preds.append(m.predict_proba(X_val_tab)[:, 1])
        all_test_preds.append(m.predict_proba(X_test_tab)[:, 1])

    final_val_preds = np.mean(all_val_preds, axis=0)
    final_test_preds = np.mean(all_test_preds, axis=0)
    
    print(f"Hybrid Ensemble Validation AUC: {roc_auc_score(y_val, final_val_preds):.4f}")
    return final_val_preds, final_test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "hybrid_pseudo_ensemble": train_hybrid_pseudo_label_ensemble,
}
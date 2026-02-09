import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import DistilBertForTokenClassification, AdamW
import pandas as pd
import numpy as np
import os
import pickle
import string
from typing import Tuple, Any, Dict, Callable, List

# Task-adaptive type definitions
X = pd.DataFrame
y = np.ndarray
Predictions = np.ndarray

# Model Function Type
ModelFn = Callable[
    [X, y, X, y, X],
    Tuple[Predictions, Predictions]
]

# Constants
CLASSES = ['PLAIN', 'PUNCT', 'DATE', 'LETTERS', 'CARDINAL', 'VERBATIM', 'MEASURE', 'ORDINAL', 'DECIMAL', 'MONEY', 'DIGIT', 'ELECTRONIC', 'TELEPHONE', 'TIME', 'FRACTION', 'ADDRESS']
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}
ID2CLASS = {i: c for c, i in CLASS2ID.items()}
OUTPUT_DATA_PATH = "output/9fef8e79-9e97-4657-be88-07dd4ac6f366/1/executor/output"

# Seq2Seq Vocab
CHARS = string.printable
CHAR2ID = {c: i + 4 for i, c in enumerate(CHARS)} # 0:PAD, 1:UNK, 2:SOS, 3:EOS
ID2CHAR = {i: c for c, i in CHAR2ID.items()}
VOCAB_SIZE = len(CHARS) + 20 + len(CLASSES)

# ===== Helper Classes =====

class BERTDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

class TransformerDataset(Dataset):
    def __init__(self, before_chars, class_ids, after_chars=None, max_in=32, max_out=128):
        self.before_chars = before_chars
        self.class_ids = class_ids
        self.after_chars = after_chars
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.before_chars)

    def __getitem__(self, idx):
        class_token = self.class_ids[idx] + len(CHARS) + 4
        src = [class_token] + list(self.before_chars[idx])
        src = src[:self.max_in] + [0] * (max(0, self.max_in - len(src)))
        
        if self.after_chars is not None:
            trg_raw = self.after_chars[idx]
            trg_in = [2] + trg_raw[:self.max_out-1]
            trg_out = trg_raw[:self.max_out-1] + [3]
            trg_in = trg_in + [0] * (max(0, self.max_out - len(trg_in)))
            trg_out = trg_out + [0] * (max(0, self.max_out - len(trg_out)))
            return torch.tensor(src), torch.tensor(trg_in), torch.tensor(trg_out)
        return torch.tensor(src)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg):
        src_emb = self.pos_encoder(self.embedding(src))
        trg_emb = self.pos_encoder(self.embedding(trg))
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        out = self.transformer(src_emb, trg_emb, tgt_mask=trg_mask)
        return self.fc_out(out)

# ===== Training Logic =====

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def train_bert_worker(rank, world_size, input_ids, attention_mask, labels, num_epochs=3):
    setup_ddp(rank, world_size)
    dataset = BERTDataset(input_ids, attention_mask, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

    model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(CLASSES)).to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for i, (b_ids, b_mask, b_labels) in enumerate(loader):
            b_ids, b_mask, b_labels = b_ids.to(rank), b_mask.to(rank), b_labels.to(rank)
            optimizer.zero_grad()
            outputs = model(b_ids, attention_mask=b_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 500 == 0:
                print(f"BERT Epoch {epoch} Batch {i} Loss {loss.item():.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(OUTPUT_DATA_PATH, "bert_model.bin"))
    
    dist.barrier()
    cleanup_ddp()

def train_transformer(train_data, device, num_epochs=5):
    src_chars, class_ids, trg_chars = train_data
    dataset = TransformerDataset(src_chars, class_ids, trg_chars)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
    
    model = CharTransformer(VOCAB_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (src, trg_in, trg_out) in enumerate(loader):
            src, trg_in, trg_out = src.to(device), trg_in.to(device), trg_out.to(device)
            optimizer.zero_grad()
            output = model(src, trg_in)
            loss = criterion(output.view(-1, VOCAB_SIZE), trg_out.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Transformer Epoch {epoch} Loss {total_loss/len(loader):.4f}")
    return model

def predict_bert(X_df, device_id):
    device = torch.device(f"cuda:{device_id}")
    model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(CLASSES)).to(device)
    model_path = os.path.join(OUTPUT_DATA_PATH, "bert_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df_temp = X_df.copy()
    df_temp['sent_key'] = df_temp['input_ids'].apply(tuple)
    unique_sents = df_temp.drop_duplicates('sent_key')
    input_ids = np.stack(unique_sents['input_ids'].values)
    attn_masks = np.stack(unique_sents['attention_mask'].values)
    
    batch_size = 64
    all_preds_raw = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            b_ids = torch.tensor(input_ids[i:i+batch_size]).to(device)
            b_mask = torch.tensor(attn_masks[i:i+batch_size]).to(device)
            logits = model(b_ids, attention_mask=b_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds_raw.append(preds)
    
    all_preds_raw = np.concatenate(all_preds_raw, axis=0) 
    sent_key_to_preds = {key: preds for key, preds in zip(unique_sents['sent_key'], all_preds_raw)}
    
    token_classes = []
    for _, row in df_temp.iterrows():
        preds = sent_key_to_preds[row['sent_key']]
        ws = row['word_start']
        if 0 <= ws < 128:
            token_classes.append(ID2CLASS[preds[ws]])
        else:
            token_classes.append('PLAIN')
    return token_classes

def predict_transformer(model, tokens_to_predict, device):
    model.eval()
    src_chars = [t[0] for t in tokens_to_predict]
    class_ids = [CLASS2ID[t[1]] for t in tokens_to_predict]
    dataset = TransformerDataset(src_chars, class_ids)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    results = []
    with torch.no_grad():
        for src in loader:
            src = src.to(device)
            trg = torch.zeros((src.size(0), 1), dtype=torch.long).to(device).fill_(2)
            for _ in range(64):
                out = model(src, trg)
                next_token = torch.argmax(out[:, -1:, :], dim=-1)
                trg = torch.cat([trg, next_token], dim=1)
                if (next_token == 3).all(): break
            
            for seq in trg.cpu().numpy():
                chars = []
                for idx in seq[1:]:
                    if idx == 3: break
                    chars.append(ID2CHAR.get(idx, ''))
                results.append("".join(chars))
    return results

# ===== Main Training Function =====

def train_bert_transformer_dual(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[Predictions, Predictions]:
    print("Preparing BERT sentence-level data via surrogate IDs...")
    def get_bert_inputs(df):
        df_temp = df.copy()
        df_temp['sent_key'] = df_temp['input_ids'].apply(tuple)
        sent_groups = df_temp.groupby('sent_key', sort=False)
        
        input_ids_list, masks_list, labels_list = [], [], []
        for _, group in sent_groups:
            input_ids_list.append(np.array(group['input_ids'].iloc[0]))
            masks_list.append(np.array(group['attention_mask'].iloc[0]))
            lbl = np.full(128, -100)
            for _, row in group.iterrows():
                ws = row['word_start']
                if 0 <= ws < 128:
                    lbl[ws] = CLASS2ID.get(row['class'], 0)
            labels_list.append(lbl)
        return np.array(input_ids_list), np.array(masks_list), np.array(labels_list)

    b_input_ids, b_masks, b_labels = get_bert_inputs(X_train)
    
    print("Training BERT Token Classifier on 2 GPUs...")
    mp.spawn(train_bert_worker, args=(2, b_input_ids, b_masks, b_labels, 3), nprocs=2, join=True)

    norm_train = X_train[~X_train['class'].isin(['PLAIN', 'PUNCT'])]
    norm_y = y_train[norm_train.index]
    src_chars = norm_train['char_seq'].tolist()
    class_ids = norm_train['class'].map(CLASS2ID).tolist()
    trg_chars = [[CHAR2ID.get(c, 1) for c in str(s)] for s in norm_y]
    
    print("Training Transformer Normalizer on GPU 1...")
    transformer_model = train_transformer((src_chars, class_ids, trg_chars), torch.device("cuda:1"), num_epochs=5)

    print("Running Inference...")
    val_classes = predict_bert(X_val, device_id=0)
    test_classes = predict_bert(X_test, device_id=0)
    
    lookup_path = os.path.join(OUTPUT_DATA_PATH, "lookup_map.pkl")
    lookup_map = {}
    if os.path.exists(lookup_path):
        with open(lookup_path, 'rb') as f:
            lookup_map = pickle.load(f)

    def assemble_predictions(df, predicted_classes, transformer_model):
        preds = [None] * len(df)
        to_gen = []
        gen_indices = []
        for i in range(len(df)):
            p_class = predicted_classes[i]
            before = str(df['before'].iloc[i])
            if p_class in ['PLAIN', 'PUNCT']:
                preds[i] = before
            elif (before, p_class) in lookup_map:
                preds[i] = lookup_map[(before, p_class)]
            else:
                to_gen.append((df['char_seq'].iloc[i], p_class))
                gen_indices.append(i)
        
        if to_gen:
            gen_texts = predict_transformer(transformer_model, to_gen, torch.device("cuda:1"))
            for i, text in enumerate(gen_texts):
                preds[gen_indices[i]] = text
        return np.array([p if p is not None else str(df['before'].iloc[i]) for i, p in enumerate(preds)])

    val_preds = assemble_predictions(X_val, val_classes, transformer_model)
    test_preds = assemble_predictions(X_test, test_classes, transformer_model)

    return val_preds, test_preds

# ===== Model Registry =====
PREDICTION_ENGINES: Dict[str, ModelFn] = {
    "bert_transformer_dual": train_bert_transformer_dual,
}
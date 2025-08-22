#!/usr/bin/env python3
"""
VAE-NCF faithful model implementation + 80/20 train/test evaluation protocol.
Calculates MAE, RMSE, HR@{5,10}, and NDCG@{5,10}.
"""
import os
import argparse
import random
from typing import List, Tuple, Dict
import heapq

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import gzip
import json

EPS = 1e-8

# ---------------------------
# Utilities & reproducibility
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------
# Data loading & splitting
# ---------------------------
def load_data(dataset_name: str, data_dir: str = "../data", subset_size: int = None, k_core: int = 5):
    """
    Returns a single dataframe with columns: user_id, item_id, rating, timestamp
    Remaps ids to contiguous integers.
    """
    print(f"Loading dataset {dataset_name} ...")
    if 'ml-' in dataset_name:
        if dataset_name == 'ml-100k':
            path = os.path.join(data_dir, 'ml-100k', 'u.data')
            df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        elif dataset_name == 'ml-1m':
            path = os.path.join(data_dir, 'ml-1m', 'ratings.dat')
            df = pd.read_csv(path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
        else:
            raise ValueError("Unsupported ml dataset")
        df['user_id'] -= 1
        df['item_id'] -= 1

    elif dataset_name == 'amazon-books':
        path = os.path.join(data_dir, 'amazon-books', 'reviews_Books_5.json.gz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Amazon Books file not found at {path}")
        print(f"Streaming Amazon subset (subset_size={subset_size}) ...")
        data = []
        with gzip.open(path, 'rb') as f:
            for i, line in enumerate(tqdm(f, total=subset_size if subset_size else None)):
                if subset_size and i >= subset_size: break
                j = json.loads(line)
                if 'reviewerID' not in j or 'asin' not in j: continue
                data.append({'user_id': j['reviewerID'], 'item_id': j['asin'], 'rating': j.get('overall', 1.0), 'timestamp': j.get('unixReviewTime', 0)})
        df = pd.DataFrame(data)

    elif dataset_name == 'lastfm':
        path = os.path.join(data_dir, 'lastfm', 'user_artists.dat')
        df = pd.read_csv(path, sep='\t', names=['user_id','item_id','weight'])
        df['timestamp'] = df.index
        df = df.drop(columns=['weight'])
    else:
        raise ValueError("Unsupported dataset")

    if k_core > 0:
        while True:
            ucounts = df['user_id'].value_counts()
            icounts = df['item_id'].value_counts()
            before = len(df)
            df = df[df['user_id'].isin(ucounts[ucounts >= k_core].index)]
            df = df[df['item_id'].isin(icounts[icounts >= k_core].index)]
            after = len(df)
            if after == before:
                break

    df['rating'] = 1.0
    df['user_id'] = df['user_id'].astype('category').cat.codes
    df['item_id'] = df['item_id'].astype('category').cat.codes
    print(f"Loaded {len(df)} interactions; users={df['user_id'].nunique()}, items={df['item_id'].nunique()}")
    return df

def per_user_split(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio_within_train: float = 0.1, seed: int = 42):
    """
    Per-user random 80/20 split of interactions into train/test.
    Then from each user's train interactions carve out val_ratio_within_train fraction into validation.
    """
    np.random.seed(seed)
    train_list, val_list, test_list = [], [], []
    min_interactions = 5
    for uid, g in df.groupby('user_id'):
        if len(g) < min_interactions:
            train_list.append(g)
            continue
        idx = g.index.values.copy()
        np.random.shuffle(idx)
        n_train = int(np.floor(len(idx) * train_ratio))
        if n_train < 1:
            train_idx, test_idx = idx[:-1], idx[-1:]
        else:
            train_idx, test_idx = idx[:n_train], idx[n_train:]
        if len(train_idx) >= 2 and val_ratio_within_train > 0:
            n_val = max(1, int(np.ceil(len(train_idx) * val_ratio_within_train)))
            if n_val >= len(train_idx):
                n_val = 1
            val_idx = np.random.choice(train_idx, size=n_val, replace=False)
            train_idx = np.setdiff1d(train_idx, val_idx)
            val_list.append(df.loc[val_idx])
        if len(train_idx) > 0:
            train_list.append(df.loc[train_idx])
        if len(test_idx) > 0:
            test_list.append(df.loc[test_idx])

    train_df = pd.concat(train_list) if train_list else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_list) if val_list else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=df.columns)
    print(f"Split => train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df

# ---------------------------
# Negative sampling utilities
# ---------------------------
def sample_negatives_for_training(train_df: pd.DataFrame, num_users: int, num_items: int, num_negatives: int, seed: int = 42):
    """
    For each positive (u,i) in train_df, sample num_negatives negative items.
    """
    rng = np.random.RandomState(seed)
    positives = list(zip(train_df['user_id'].tolist(), train_df['item_id'].tolist()))
    neg_pairs = []
    user_pos = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    for (u, _) in positives:
        u_items = user_pos.get(u, set())
        negs = []
        while len(negs) < num_negatives:
            cand = int(rng.randint(0, num_items))
            if cand not in u_items:
                negs.append((u, cand))
        neg_pairs.extend(negs)
    user_item_pairs = positives + neg_pairs
    labels = [1.0] * len(positives) + [0.0] * len(neg_pairs)
    return user_item_pairs, labels

# ---------------------------
# Dataset class used during training
# ---------------------------
class TrainingDataset(Dataset):
    def __init__(self, user_item_pairs: List[Tuple[int,int]], labels: List[float], user_interaction_matrix: csr_matrix, item_interaction_matrix: csr_matrix):
        self.pairs = user_item_pairs
        self.labels = labels
        self.user_matrix = user_interaction_matrix
        self.item_matrix = item_interaction_matrix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        label = self.labels[idx]
        uvec = torch.from_numpy(self.user_matrix[u].toarray().squeeze()).float()
        ivec = torch.from_numpy(self.item_matrix[i].toarray().squeeze()).float()
        return uvec, ivec, torch.tensor(label, dtype=torch.float32)

# ---------------------------
# Model: VAE & VAE_NCF
# ---------------------------
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, output_activation: str = 'sigmoid'):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc.append(nn.Linear(prev, h))
            enc.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.append(nn.Linear(prev, h))
            dec.append(nn.ReLU())
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        if output_activation == 'sigmoid':
            dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z

class VAE_NCF(nn.Module):
    def __init__(self, num_users: int, num_items: int, vae_hidden_dims: List[int], latent_dim: int, mlp_layers: List[int], dropout: float = 0.2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.user_vae = VAE(input_dim=num_items, hidden_dims=vae_hidden_dims, latent_dim=latent_dim)
        self.item_vae = VAE(input_dim=num_users, hidden_dims=vae_hidden_dims, latent_dim=latent_dim)
        mlp_input_dim = 2 * latent_dim
        modules = []
        for h in mlp_layers:
            modules.append(nn.Linear(mlp_input_dim, h))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            mlp_input_dim = h
        self.mlp = nn.Sequential(*modules)
        gmf_dim = latent_dim
        pred_in_dim = mlp_layers[-1] + gmf_dim
        self.prediction_layer = nn.Linear(pred_in_dim, 1)

    def forward(self, user_interactions, item_interactions):
        user_recon, user_mu, user_logvar, user_z = self.user_vae(user_interactions)
        item_recon, item_mu, item_logvar, item_z = self.item_vae(item_interactions)
        gmf_vec = user_z * item_z
        mlp_in = torch.cat([user_z, item_z], dim=-1)
        mlp_vec = self.mlp(mlp_in)
        ncf_vec = torch.cat([gmf_vec, mlp_vec], dim=-1)
        logit = self.prediction_layer(ncf_vec).squeeze(-1)
        return {
            'logit': logit,
            'user_recon': user_recon, 'user_mu': user_mu, 'user_logvar': user_logvar,
            'item_recon': item_recon, 'item_mu': item_mu, 'item_logvar': item_logvar
        }

# ---------------------------
# Loss calculation
# ---------------------------
def compute_losses(model_out: Dict, user_interactions: torch.Tensor, item_interactions: torch.Tensor, labels: torch.Tensor):
    l_ncf = F.binary_cross_entropy_with_logits(model_out['logit'], labels)
    user_recon_loss = F.mse_loss(model_out['user_recon'], user_interactions, reduction='sum')
    item_recon_loss = F.mse_loss(model_out['item_recon'], item_interactions, reduction='sum')
    recon_loss = user_recon_loss + item_recon_loss
    user_kld = -0.5 * torch.sum(1 + model_out['user_logvar'] - model_out['user_mu'].pow(2) - model_out['user_logvar'].exp())
    item_kld = -0.5 * torch.sum(1 + model_out['item_logvar'] - model_out['item_mu'].pow(2) - model_out['item_logvar'].exp())
    kld_loss = user_kld + item_kld
    batch_size = user_interactions.shape[0]
    l_vae = (recon_loss + kld_loss) / (batch_size + EPS)
    return l_ncf, l_vae

# ---------------------------
# Train & validation loops
# ---------------------------
def train_model(model: VAE_NCF,
                train_loader: DataLoader,
                val_train_matrix: csr_matrix,
                val_test_matrix: csr_matrix,
                optimizer: torch.optim.Optimizer,
                epochs: int,
                alpha: float,
                device: torch.device,
                early_stop_patience: int = 5,
                top_k_for_es: int = 10):
    best_val = -np.inf
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for uvec_batch, ivec_batch, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}"):
            uvec_batch, ivec_batch, labels = uvec_batch.to(device), ivec_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(uvec_batch, ivec_batch)
            l_ncf, l_vae = compute_losses(out, uvec_batch, ivec_batch, labels)
            loss = l_ncf + alpha * l_vae
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * uvec_batch.shape[0]
        avg_loss = total_loss / (len(train_loader.dataset) + EPS)
        print(f"Epoch {epoch} Train loss: {avg_loss:.6f}")

        # For early stopping, we only need one metric, e.g., NDCG@10
        val_metrics = evaluate_all_metrics(model, val_train_matrix, val_test_matrix, [top_k_for_es], device=device, eval_batch_items=1024)
        val_ndcg = val_metrics.get(f'NDCG@{top_k_for_es}', 0.0)
        print(f"Validation NDCG@{top_k_for_es}: {val_ndcg:.6f}")

        if val_ndcg > best_val + 1e-6:
            best_val = val_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            print("New best validation score. Saving model state.")
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping triggered (patience={early_stop_patience}). Restoring best model.")
                if best_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                break
    return model

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_all_metrics(model: VAE_NCF,
                         train_matrix: csr_matrix,
                         test_matrix: csr_matrix,
                         top_k_list: List[int],
                         device: torch.device,
                         eval_batch_items: int = 1024) -> Dict[str, float]:
    model.eval()
    n_users, n_items = train_matrix.shape
    hits, ndcgs = {k: [] for k in top_k_list}, {k: [] for k in top_k_list}
    item_matrix = train_matrix.transpose().tocsr()
    
    test_users = np.unique(test_matrix.nonzero()[0])
    
    # --- Ranking Metrics (HR, NDCG) ---
    with torch.no_grad():
        for u in tqdm(test_users, desc="Eval Ranking"):
            true_items = set(test_matrix[u].nonzero()[1].tolist())
            if not true_items:
                continue
            uvec = torch.from_numpy(train_matrix[u].toarray().squeeze()).float().to(device)
            scores = np.zeros(n_items, dtype=np.float32)
            start = 0
            while start < n_items:
                end = min(n_items, start + eval_batch_items)
                items_idx = np.arange(start, end)
                ivecs = torch.from_numpy(item_matrix[items_idx].toarray()).float().to(device)
                us_repeated = uvec.unsqueeze(0).repeat(ivecs.shape[0], 1)
                logits = model(us_repeated, ivecs)['logit'].cpu().numpy()
                scores[start:end] = logits
                start = end
            train_items = set(train_matrix[u].nonzero()[1].tolist())
            if train_items:
                scores[list(train_items)] = -np.inf
            ranked_indices = np.argsort(-scores)
            for k in top_k_list:
                topk = ranked_indices[:k]
                hits[k].append(1.0 if len(set(topk).intersection(true_items)) > 0 else 0.0)
                idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(k, len(true_items)))])
                dcg = np.sum([1.0 / np.log2(i + 2) for i, item in enumerate(topk) if item in true_items])
                ndcgs[k].append(dcg / (idcg + EPS))

    results = {}
    for k in top_k_list:
        results[f'HR@{k}'] = float(np.mean(hits[k])) if hits[k] else 0.0
        results[f'NDCG@{k}'] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0

    # --- Prediction Metrics (MAE, RMSE) ---
    test_user_item_pairs = list(zip(*test_matrix.nonzero()))
    test_labels = [1.0] * len(test_user_item_pairs)
    test_preds = []
    with torch.no_grad():
        for u, i in tqdm(test_user_item_pairs, desc="Eval Prediction"):
            uvec = torch.from_numpy(train_matrix[u].toarray()).float().to(device)
            ivec = torch.from_numpy(item_matrix[i].toarray()).float().to(device)
            logit = model(uvec, ivec)['logit'].cpu().item()
            pred_prob = torch.sigmoid(torch.tensor(logit)).item()
            test_preds.append(pred_prob)

    results['MAE'] = mean_absolute_error(test_labels, test_preds)
    results['RMSE'] = np.sqrt(mean_squared_error(test_labels, test_preds))
    
    return results

# ---------------------------
# Main: training & eval wiring
# ---------------------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_data(args.dataset, data_dir=args.data_dir, subset_size=args.subset_size, k_core=args.k_core)
    num_users, num_items = df['user_id'].nunique(), df['item_id'].nunique()

    train_df, val_df, test_df = per_user_split(df, train_ratio=0.8, val_ratio_within_train=0.1, seed=args.seed)

    train_matrix = csr_matrix((np.ones(len(train_df)), (train_df['user_id'], train_df['item_id'])), shape=(num_users, num_items))
    val_train_matrix = csr_matrix((np.ones(len(train_df)), (train_df['user_id'], train_df['item_id'])), shape=(num_users, num_items))
    val_test_matrix = csr_matrix((np.ones(len(val_df)), (val_df['user_id'], val_df['item_id'])), shape=(num_users, num_items))
    test_matrix = csr_matrix((np.ones(len(test_df)), (test_df['user_id'], test_df['item_id'])), shape=(num_users, num_items))
    item_matrix = train_matrix.transpose().tocsr()

    train_pairs, train_labels = sample_negatives_for_training(train_df, num_users, num_items, args.num_negatives, seed=args.seed)
    train_dataset = TrainingDataset(train_pairs, train_labels, train_matrix, item_matrix)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model = VAE_NCF(num_users=num_users, num_items=num_items,
                    vae_hidden_dims=[int(x) for x in args.vae_dims.split(',')] if args.vae_dims else [64, 32],
                    latent_dim=args.latent_dim,
                    mlp_layers=[int(x) for x in args.mlp_layers.split(',')],
                    dropout=args.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print("Starting main training with early stopping based on validation NDCG@10 ...")
    model = train_model(model, train_loader, val_train_matrix, val_test_matrix, optimizer,
                        args.epochs, args.alpha, device,
                        early_stop_patience=args.early_stop_patience, top_k_for_es=10)
    
    # --- Final Comprehensive Evaluation ---
    final_top_k_list = [5, 10]
    results = evaluate_all_metrics(model, train_matrix, test_matrix, final_top_k_list, device=device, eval_batch_items=args.eval_batch_items)
    
    print("\n" + "="*30)
    print("      Final Test Metrics")
    print("="*30)
    print(f"  MAE:        {results['MAE']:.4f}")
    print(f"  RMSE:       {results['RMSE']:.4f}")
    print("-" * 30)
    print(f"  HR@5:       {results['HR@5']:.4f}")
    print(f"  NDCG@5:     {results['NDCG@5']:.4f}")
    print("-" * 30)
    print(f"  HR@10:      {results['HR@10']:.4f}")
    print(f"  NDCG@10:    {results['NDCG@10']:.4f}")
    print("="*30 + "\n")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', choices=['ml-100k','ml-1m','amazon-books','lastfm'])
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--k_core', type=int, default=5)

    # architecture
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--vae_dims', type=str, default='128,64')
    parser.add_argument('--mlp_layers', type=str, default='64,32,16,8')
    parser.add_argument('--dropout', type=float, default=0.2)

    # training
    parser.add_argument('--alpha', type=float, default=0.01, help="Weight for the VAE loss term")
    parser.add_argument('--num_negatives', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    # eval / early stop
    parser.add_argument('--eval_batch_items', type=int, default=1024)
    parser.add_argument('--early_stop_patience', type=int, default=5)

    args = parser.parse_args()
    main(args)
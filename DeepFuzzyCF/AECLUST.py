import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import gzip
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SparseDataset(Dataset):
    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix
        self.n_users = sparse_matrix.shape[0]

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        user_vector = self.sparse_matrix[idx].toarray().squeeze()
        return torch.FloatTensor(user_vector), idx


def load_data(dataset_name='ml-100k', data_dir='./', train_ratio=0.8, k_core=5, max_reviews=None, val_ratio=0.1):
    """
    Loads dataset (ml-100k, ml-1m, amazon-books (jsonl.gz subset), lastfm).
    Performs k-core filtering, remaps user/item ids, performs per-user 80/20 split into train/test,
    then splits train further into train/val (val_ratio fraction per user).
    Returns: train_matrix, val_matrix, test_matrix, n_users, n_items (all csr_matrix)
    """
    print(f"\nLoading and processing {dataset_name} dataset...")
    if 'ml-' in dataset_name:
        if dataset_name == 'ml-100k':
            data_path = os.path.join(data_dir, 'ml-100k', 'u.data')
            df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        elif dataset_name == 'ml-1m':
            data_path = os.path.join(data_dir, 'ml-1m', 'ratings.dat')
            df = pd.read_csv(data_path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
        df['user_id'] -= 1
        df['item_id'] -= 1
    elif dataset_name == 'amazon-books':
        data_path = os.path.join(data_dir, 'amazon-books', 'Books.jsonl.gz')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at {data_path}.")
        print(f"Streaming a subset of data (max_reviews={max_reviews})...")
        data = []
        with gzip.open(data_path, 'rb') as f:
            for i, line in enumerate(tqdm(f, desc="Reading Amazon subset", total=max_reviews if max_reviews else None)):
                if max_reviews and i >= max_reviews:
                    break
                review = json.loads(line)
                # Some reviews may not have rating or user id; guard
                if 'user_id' not in review or 'asin' not in review:
                    continue
                rating = review.get('rating', 1.0)
                data.append({'user_id': review['user_id'], 'item_id': review['asin'], 'rating': rating})
        df = pd.DataFrame(data)
        print(f"Loaded {df.shape[0]} reviews.")
        print(f"Applying {k_core}-core filtering on the subset...")
        if k_core > 0:
            while True:
                user_counts = df['user_id'].value_counts()
                item_counts = df['item_id'].value_counts()
                initial_rows = df.shape[0]
                df = df[df['user_id'].isin(user_counts[user_counts >= k_core].index)]
                df = df[df['item_id'].isin(item_counts[item_counts >= k_core].index)]
                final_rows = df.shape[0]
                if initial_rows == final_rows:
                    break
    elif dataset_name == 'lastfm':
        data_path = os.path.join(data_dir, 'lastfm', 'user_artists.dat')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at {data_path}.")
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'weight'])
        df = df.drop(columns=['weight'])
        df.columns = ['user_id', 'item_id']
        print(f"Applying {k_core}-core filtering for Last.fm...")
        if k_core > 0:
            while True:
                user_counts = df['user_id'].value_counts()
                item_counts = df['item_id'].value_counts()
                initial_rows = df.shape[0]
                df = df[df['user_id'].isin(user_counts[user_counts >= k_core].index)]
                df = df[df['item_id'].isin(item_counts[item_counts >= k_core].index)]
                final_rows = df.shape[0]
                if initial_rows == final_rows:
                    break
    else:
        raise ValueError("Dataset not supported.")

    if df.empty:
        raise ValueError(f"DataFrame is empty after {k_core}-core filtering. Try a smaller k_core value or a larger subset_size.")

    # Treat everything as implicit binary interactions
    df['rating'] = 1.0
    # remap ids to contiguous categories
    df['user_id'] = df['user_id'].astype('category').cat.codes
    df['item_id'] = df['item_id'].astype('category').cat.codes
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    print(f"Found {n_users} users and {n_items} items after filtering and remapping.")

    # per-user train/test split (only users with >=5 interactions considered)
    train_data, test_data = [], []
    for _, user_ratings in tqdm(df.groupby('user_id'), desc="Splitting data"):
        if len(user_ratings) < 5:
            # skip users with too few interactions to create reliable train/test
            continue
        train_indices = np.random.choice(user_ratings.index, size=int(len(user_ratings) * train_ratio), replace=False)
        test_indices = list(set(user_ratings.index) - set(train_indices))
        train_data.append(user_ratings.loc[train_indices])
        if test_indices:
            test_data.append(user_ratings.loc[test_indices])

    if not train_data:
        raise ValueError("No training data generated after splitting. Check user interaction counts.")
    train_df, test_df = pd.concat(train_data), pd.concat(test_data)

    # Now split train_df into train + val (val_ratio per-user)
    train_records = []
    val_records = []
    for user, user_ratings in train_df.groupby('user_id'):
        n = len(user_ratings)
        if n == 0:
            continue
        val_count = int(np.ceil(n * val_ratio)) if n >= 2 else 0
        if val_count > 0:
            val_indices = np.random.choice(user_ratings.index, size=val_count, replace=False)
            val_records.append(user_ratings.loc[val_indices])
            train_records.append(user_ratings.drop(val_indices))
        else:
            train_records.append(user_ratings)

    train_df_final = pd.concat(train_records) if train_records else pd.DataFrame(columns=train_df.columns)
    val_df = pd.concat(val_records) if val_records else pd.DataFrame(columns=train_df.columns)

    def create_sparse_matrix(data, n_users, n_items):
        if data.empty:
            return csr_matrix((n_users, n_items), dtype=np.float32)
        rows = data['user_id'].values
        cols = data['item_id'].values
        values = data['rating'].values
        return csr_matrix((values, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    train_matrix = create_sparse_matrix(train_df_final, n_users, n_items)
    val_matrix = create_sparse_matrix(val_df, n_users, n_items)
    test_matrix = create_sparse_matrix(test_df, n_users, n_items)
    sparsity = 1 - (df.shape[0] / (n_users * n_items))
    print(f"Train interactions: {train_df_final.shape[0]}, Val interactions: {val_df.shape[0]}, Test interactions: {test_df.shape[0]}")
    print(f"Overall data sparsity: {sparsity:.4f}")

    return train_matrix, val_matrix, test_matrix, n_users, n_items


def kl_divergence_sparsity(activations, rho):
    rho_hat = torch.mean(activations, 0)
    rho_tensor = torch.tensor([rho] * len(rho_hat), device=device)
    rho_hat = torch.clamp(rho_hat, 1e-8, 1 - 1e-8)
    kl = rho_tensor * torch.log(rho_tensor / rho_hat) + (1 - rho_tensor) * torch.log((1 - rho_tensor) / (1 - rho_hat))
    return torch.sum(kl)


class DeepFuzzyClusterCF(nn.Module):
    def __init__(self, n_items, n_clusters, encoder_dims, alpha, rho, gamma):
        super(DeepFuzzyClusterCF, self).__init__()
        self.n_items, self.n_clusters = n_items, n_clusters
        self.encoder_dims, self.latent_dim = encoder_dims, encoder_dims[-1]
        self.alpha, self.rho, self.gamma = alpha, rho, gamma

        encoder_layers = []
        all_dims = [n_items] + encoder_dims
        # build encoder: Linear, Sigmoid alternating (like original)
        for i in range(len(all_dims) - 1):
            encoder_layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            encoder_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder: mirror
        decoder_layers = []
        reversed_dims = list(reversed(all_dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
            if i < len(reversed_dims) - 2:
                decoder_layers.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*decoder_layers)

        # cluster centers in latent space
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, self.latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, x):
        hidden_activations = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            if isinstance(layer, nn.Sigmoid):
                hidden_activations.append(h)
        z = h
        x_hat = self.decoder(z)

        # DEC-style soft assignment: q_j = (1 / (1 + dist)) normalized
        dist = torch.sum(torch.square(z.unsqueeze(1) - self.cluster_centers), dim=2)
        q_numerator = 1.0 / (1.0 + dist)
        q = q_numerator / torch.sum(q_numerator, dim=1, keepdim=True)

        return x_hat, z, hidden_activations, q

    def loss(self, x_original, x_reconstructed, hidden_activations, q, p_target, is_finetuning=False):
        reconstruction_loss = F.mse_loss(x_reconstructed, x_original)
        sparsity_loss = torch.tensor(0.0, device=device)
        if self.alpha > 0 and self.rho > 0:
            sparsity_loss = sum(kl_divergence_sparsity(h, self.rho) for h in hidden_activations)
        lr_loss = reconstruction_loss + self.alpha * sparsity_loss
        if not is_finetuning:
            return lr_loss
        clustering_loss = F.kl_div(torch.log(q + 1e-8), p_target, reduction='batchmean')
        total_loss = lr_loss + self.gamma * clustering_loss
        return total_loss, lr_loss, clustering_loss


def pretrain_model(model, train_loader, epochs, lr, dropout_rate):
    """
    Optional global pretraining (uses SGD to be consistent with paper).
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_data, _ in train_loader:
            inputs = batch_data.to(device)
            corrupted_inputs = F.dropout(inputs, p=dropout_rate, training=model.training)
            optimizer.zero_grad()
            x_hat, _, hidden_activations, _ = model(corrupted_inputs)
            loss = model.loss(inputs, x_hat, hidden_activations, None, None, is_finetuning=False)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Global pretrain] Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.6f}")


def initialize_clusters(model, full_train_data_loader):
    model.eval()
    all_z_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm(full_train_data_loader, desc="Extracting latent vectors for KMeans"):
            inputs = batch_data.to(device)
            # encoder produces latent vector after forward pass
            z = model.encoder(inputs)
            all_z_list.append(z.cpu())
    all_z = torch.cat(all_z_list, dim=0).numpy()
    kmeans = KMeans(n_clusters=model.n_clusters, n_init='auto', random_state=42)
    kmeans.fit(all_z)
    model.cluster_centers.data = torch.from_numpy(kmeans.cluster_centers_).to(device)


def compute_target_distribution(model, full_train_data_loader):
    """
    DEC-style: p = q^2 / sum_j q_j ; then row-normalize
    """
    model.eval()
    all_q_list = []
    with torch.no_grad():
        for batch_data, _ in full_train_data_loader:
            inputs = batch_data.to(device)
            _, _, _, q = model(inputs)
            all_q_list.append(q)
    all_q = torch.cat(all_q_list, dim=0)
    p = all_q**2 / torch.sum(all_q, dim=0)
    p = (p.T / torch.sum(p, dim=1)).T
    return p.detach()


def finetune_model(model, train_loader, full_train_data_loader, val_loader, epochs, lr, dropout_rate, patience=5):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        p_target = compute_target_distribution(model, full_train_data_loader)
        epoch_loss = 0.0
        for batch_data, indices in train_loader:
            inputs = batch_data.to(device)
            corrupted_inputs = F.dropout(inputs, p=dropout_rate, training=model.training)
            optimizer.zero_grad()
            x_hat, _, hidden_activations, q = model(corrupted_inputs)
            p_batch = p_target[indices].to(device)
            loss, _, _ = model.loss(inputs, x_hat, hidden_activations, q, p_batch, is_finetuning=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)

        # validation reconstruction + sparsity (no clustering term) for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch, _ in val_loader:
                v_in = val_batch.to(device)
                v_hat, _, v_hidden, _ = model(v_in)
                recon = F.mse_loss(v_hat, v_in, reduction='sum')
                sp = sum(kl_divergence_sparsity(h, model.rho) for h in v_hidden) if model.alpha > 0 else torch.tensor(0.0, device=device)
                val_loss += (recon + model.alpha * sp).item()
        val_loss /= max(1, len(val_loader.dataset))

        print(f"[Finetune] Epoch {epoch+1}/{epochs} TrainLoss: {epoch_loss:.6f} ValLoss: {val_loss:.6f}")

        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered (patience={patience}). Restoring best model.")
                if best_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                break


# Layer-wise single-layer AE for greedy pretraining
class SingleLayerAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, rho=0.0, use_sigmoid=True):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid() if use_sigmoid else nn.ReLU(True)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.rho = rho

    def forward(self, x):
        h = self.activation(self.encoder(x))
        x_hat = self.decoder(h)
        return x_hat, h

    def sparsity_loss(self, h):
        if self.rho <= 0.0:
            return torch.tensor(0.0, device=h.device)
        return kl_divergence_sparsity(h, self.rho)


def layerwise_pretrain(model, train_matrix_sparse, epochs_per_layer=10, lr=1e-3, batch_size=256, rho=0.05, momentum=0.9):
    """
    Greedy layer-wise pretraining: for each encoder layer, train a shallow AE to reconstruct the input
    to that layer. After training each layer, copy encoder weights into the stacked model and transform
    inputs for the next layer.
    """
    model.eval()
    # dense user matrix
    X = torch.FloatTensor(train_matrix_sparse.toarray()).to(device)
    cur_inputs = X.clone().detach()

    # collect linear encoder layers in order (model.encoder interleaves Linear + Sigmoid)
    encoder_layers = [l for l in model.encoder if isinstance(l, nn.Linear)]

    for layer_idx, enc_layer in enumerate(encoder_layers):
        in_dim = enc_layer.in_features
        out_dim = enc_layer.out_features
        print(f"\nLayer-wise pretraining: layer {layer_idx} (in={in_dim}, out={out_dim})")

        ae = SingleLayerAE(in_dim, out_dim, rho=rho, use_sigmoid=True).to(device)
        optimizer = optim.SGD(ae.parameters(), lr=lr, momentum=momentum)
        dataset = torch.utils.data.TensorDataset(cur_inputs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs_per_layer):
            ae.train()
            epoch_loss = 0.0
            for (batch_x, ) in loader:
                optimizer.zero_grad()
                x_hat, h = ae(batch_x)
                recon = F.mse_loss(x_hat, batch_x)
                sp = ae.sparsity_loss(h)
                loss = recon + (rho * sp if rho > 0 else 0.0)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(loader.dataset)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Layer {layer_idx} Epoch {epoch+1}/{epochs_per_layer} - Loss: {epoch_loss:.6f}")

        # copy weights to stacked model
        with torch.no_grad():
            enc_layer.weight.copy_(ae.encoder.weight.data)
            enc_layer.bias.copy_(ae.encoder.bias.data)

        # produce inputs for next layer
        ae.eval()
        with torch.no_grad():
            cur_inputs = ae.activation(ae.encoder(cur_inputs))


def evaluate_model(model, train_matrix_sparse, test_matrix_sparse, top_k_list, n_neighbors):
    """
    Evaluates model using Cosine similarity on the LEARNED LATENT VECTORS (z)
    and the prediction formula adapted from Eq.14.
    """
    model.eval()
    n_users, n_items = train_matrix_sparse.shape

    # Data loader for full train (to extract z and q)
    full_train_dataset = SparseDataset(train_matrix_sparse)
    full_train_loader = DataLoader(full_train_dataset, batch_size=256, shuffle=False)

    all_z_list, all_q_list = [], []
    with torch.no_grad():
        for batch_data, _ in tqdm(full_train_loader, desc="[Eval] Extracting latent/cluster data"):
            inputs = batch_data.to(device)
            _, z, _, q = model(inputs)
            all_z_list.append(z.cpu())
            all_q_list.append(q.cpu())

    all_z = torch.cat(all_z_list, dim=0).numpy()
    all_q = torch.cat(all_q_list, dim=0).numpy()
    if np.isnan(all_z).any():
        raise ValueError("NaN detected in latent representations (all_z).")

    # --- START: MODIFIED SECTION ---
    # This is the key change: We now compute similarity on the learned latent
    # vectors 'z', not on the original sparse rating matrix 'R'. This aligns
    # the recommendation step with the representation learning step.
    print("\nCalculating user-user similarity in the latent space (z) using Cosine Similarity...")
    similarities = cosine_similarity(all_z)
    np.fill_diagonal(similarities, 0.0) # A user is not their own neighbor
    # --- END: MODIFIED SECTION ---

    # user mean rating (implicit) as fraction of items rated (still needed for prediction formula)
    user_mean_ratings = np.array(train_matrix_sparse.sum(axis=1)).squeeze() / n_items

    # Metric accumulators
    hits, ndcgs = defaultdict(list), defaultdict(list)
    test_users = np.unique(test_matrix_sparse.nonzero()[0])

    for user_id in tqdm(test_users, desc="[Eval] Calculating metrics per user"):
        true_items = set(test_matrix_sparse[user_id].nonzero()[1])
        if not true_items:
            continue

        # Neighbor selection: allocate neighbors proportional to user's membership across clusters
        user_memberships = all_q[user_id]
        neighbor_pool = []
        num_neighbors_per_cluster = np.round(n_neighbors * user_memberships).astype(int)

        for cluster_id in range(model.n_clusters):
            num_to_sample = num_neighbors_per_cluster[cluster_id]
            if num_to_sample == 0:
                continue
            # candidate ranking: latent similarity weighted by membership of candidate in this cluster
            candidate_scores = similarities[user_id, :] * all_q[:, cluster_id]
            candidate_scores[user_id] = -np.inf
            top_candidates = np.argsort(-candidate_scores)[:num_to_sample]
            neighbor_pool.extend(top_candidates)

        if not neighbor_pool:
            continue

        neighbors = np.unique(neighbor_pool)
        # The similarities are now from the latent space
        neighbor_sims = similarities[user_id, neighbors]
        # The ratings are from the original training data
        neighbor_ratings = train_matrix_sparse[neighbors, :].toarray()

        # Prediction Eq.14 (Pearson-style) adapted for implicit and using latent similarity:
        user_x_mean = user_mean_ratings[user_id]
        neighbor_mean_ratings = user_mean_ratings[neighbors]
        # The deviation is from the mean rating in the *original* space
        deviations_matrix = neighbor_ratings - neighbor_mean_ratings.reshape(-1, 1)
        # The weighted sum uses the *new* latent similarities
        numerator = np.dot(neighbor_sims, deviations_matrix)
        denominator = np.sum(np.abs(neighbor_sims)) + 1e-8
        pred_scores = user_x_mean + (numerator / denominator)

        user_rated_items_mask = train_matrix_sparse[user_id].toarray().squeeze() > 0
        pred_scores[user_rated_items_mask] = -np.inf

        ranked_item_indices = np.argsort(pred_scores)[::-1]

        for k in top_k_list:
            top_k_items = set(ranked_item_indices[:k])
            num_hits = len(top_k_items.intersection(true_items))
            hits[k].append(1.0 if num_hits > 0 else 0.0)

            idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(k, len(true_items)))])
            hit_positions = [i for i, item in enumerate(ranked_item_indices[:k]) if item in true_items]
            dcg = np.sum([1.0 / np.log2(pos + 2) for pos in hit_positions])
            ndcgs[k].append(dcg / (idcg + 1e-8))

    results = {f'HR@{k}': np.mean(hits[k]) if hits[k] else 0.0 for k in top_k_list}
    results.update({f'NDCG@{k}': np.mean(ndcgs[k]) if ndcgs[k] else 0.0 for k in top_k_list})
    return results


def visualize_clusters(model, full_train_loader, dataset_name, results_dir):
    print("\n--- Generating Cluster Visualizations ---")
    model.eval()
    all_z_list, all_q_list = [], []
    with torch.no_grad():
        for batch_data, _ in tqdm(full_train_loader, desc="Extracting data for visualization"):
            inputs = batch_data.to(device)
            _, z, _, q = model(inputs)
            all_z_list.append(z.cpu())
            all_q_list.append(q.cpu())
    all_z = torch.cat(all_z_list, dim=0).numpy()
    all_q = torch.cat(all_q_list, dim=0).numpy()
    if np.std(all_z, axis=0).mean() < 1e-6:
        print("\nWARNING: Model has collapsed. Skipping visualization.")
        return
    hard_assignments = np.argmax(all_q, axis=1)
    print("Performing t-SNE dimensionality reduction...")
    sample_size = min(all_z.shape[0], 2500)
    user_indices = np.random.choice(all_z.shape[0], size=sample_size, replace=False)
    z_sample, assignments_sample = all_z[user_indices], hard_assignments[user_indices]
    perplexity_value = min(30, sample_size - 1)
    if perplexity_value <= 0:
        print("Skipping t-SNE: not enough samples.")
        return
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(z_sample)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    palette = sns.color_palette("husl", model.n_clusters)
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=[palette[i] for i in assignments_sample], alpha=0.7, s=30)
    ax.set_title(f't-SNE Visualization of User Clusters ({dataset_name})', fontsize=18, weight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10, label=f'Cluster {i}') for i in range(model.n_clusters)]
    ax.legend(handles=legend_handles, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plot_filename = os.path.join(results_dir, f'cluster_visualization_{dataset_name}.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Cluster visualization saved to '{plot_filename}'")
    plt.show()


def objective(trial, args, train_matrix, val_matrix, test_matrix, n_users, n_items):
    set_seed(args.seed)

    # search space (kept simplified)
    n_clusters = trial.suggest_categorical('n_clusters', [5, 10, 20])
    gamma = trial.suggest_categorical('gamma', [0.01, 0.1])
    dropout = trial.suggest_categorical('dropout', [0.3])
    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128])

    lr = args.lr
    alpha = args.alpha
    rho = args.rho
    n_neighbors = args.neighbors

    if latent_dim == 128:
        encoder_dims = [512, 256, 128]
    elif latent_dim == 32:
        encoder_dims = [128, 64, 32]
    else:
        encoder_dims = [256, 128, 64]

    train_dataset = SparseDataset(train_matrix)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    full_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataset = SparseDataset(val_matrix)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = DeepFuzzyClusterCF(n_items, n_clusters, encoder_dims, alpha, rho, gamma).to(device)

    print(f"\n--- Starting Trial {trial.number} | Params: C={n_clusters}, G={gamma}, D={dropout}, Latent={latent_dim} ---")
    # layer-wise pretraining
    layerwise_pretrain(model, train_matrix, epochs_per_layer=args.pretrain_layer_epochs, lr=args.lr, batch_size=args.batch_size, rho=args.rho, momentum=0.9)
    # optional global pretrain
    if args.pretrain_global_epochs > 0:
        pretrain_model(model, train_loader, args.pretrain_global_epochs, args.lr, dropout)

    initialize_clusters(model, full_train_loader)
    finetune_model(model, train_loader, full_train_loader, val_loader, args.finetune_epochs, args.lr / 5, dropout, patience=args.patience)

    top_k_list_for_eval = [5, 10, 15]
    results = evaluate_model(model, train_matrix, test_matrix, top_k_list_for_eval, n_neighbors)
    trial.set_user_attr("all_metrics", results)
    metric_to_optimize = results.get('NDCG@10', 0.0)
    print(f"--- Trial {trial.number} Finished --- NDCG@10: {metric_to_optimize:.4f} --- All Metrics: {results}")
    return metric_to_optimize


def run_single_trial(args, train_matrix, val_matrix, test_matrix, n_users, n_items):
    set_seed(args.seed)
    results_dir = f'results_{args.dataset}_{args.clusters}c_{args.latent_dim}d'
    os.makedirs(results_dir, exist_ok=True)

    train_dataset = SparseDataset(train_matrix)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    full_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    val_dataset = SparseDataset(val_matrix)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.latent_dim == 128:
        encoder_dims = [512, 256, 128]
    elif args.latent_dim == 64:
        encoder_dims = [256, 128, 64]
    else:
        encoder_dims = [128, 64, 32]

    model = DeepFuzzyClusterCF(n_items, args.clusters, encoder_dims, args.alpha, args.rho, args.gamma).to(device)
    print("\nModel Architecture:\n", model)

    # layer-wise pretrain
    layerwise_pretrain(model, train_matrix, epochs_per_layer=args.pretrain_layer_epochs, lr=args.lr, batch_size=args.batch_size, rho=args.rho, momentum=0.9)
    # optional global pretrain
    if args.pretrain_global_epochs > 0:
        pretrain_model(model, train_loader, args.pretrain_global_epochs, args.lr, args.dropout)

    initialize_clusters(model, full_train_loader)
    finetune_model(model, train_loader, full_train_loader, val_loader, args.finetune_epochs, args.lr / 5, args.dropout, patience=args.patience)

    top_k_list = [5, 10, 15]
    results = evaluate_model(model, train_matrix, test_matrix, top_k_list, args.neighbors)

    print("\n--- Final Results for Single Run ---")
    print(f"Dataset: {args.dataset}, Clusters: {args.clusters}, Latent Dim: {args.latent_dim}, Neighbors: {args.neighbors}, Alpha: {args.alpha}, Rho: {args.rho}")
    header = " | ".join([f"{k:<8}" for k in results.keys()])
    values = " | ".join([f"{v:<8.4f}" for v in results.values()])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(values)
    print("-" * len(header))

    visualize_clusters(model, full_train_loader, args.dataset, results_dir)


def main():
    parser = argparse.ArgumentParser(description='Deep Embedded Fuzzy Clustering for CF (modified)')
    parser.add_argument('--search', action='store_true', help='Enable hyperparameter search with Optuna.')
    parser.add_argument('--dataset', type=str, default='ml-100k', choices=['ml-100k', 'ml-1m', 'amazon-books', 'lastfm'], help='Dataset to use')
    parser.add_argument('--subset_size', type=int, default=None, help='Number of reviews to load for large datasets.')
    parser.add_argument('--k_core', type=int, default=5, help='K-core filtering for sparse datasets.')
    parser.add_argument('--clusters', type=int, default=15, help='Num clusters (single run)')
    parser.add_argument('--latent_dim', type=int, default=64, choices=[32, 64, 128], help='Latent dimension (single run)')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='(unused) old pretrain epochs param')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='Fine-tuning epochs (single run)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=1e-5, help='Sparsity regularization weight')
    parser.add_argument('--rho', type=float, default=0.05, help='Sparsity target activation')
    parser.add_argument('--gamma', type=float, default=0.01, help='Clustering loss weight')
    parser.add_argument('--neighbors', type=int, default=50, help='Num neighbors for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # new args:
    parser.add_argument('--pretrain_layer_epochs', type=int, default=10, help='Layer-wise pretraining epochs per layer')
    parser.add_argument('--pretrain_global_epochs', type=int, default=10, help='Global pretraining epochs after layer-wise pretraining (optional)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience on validation')

    args = parser.parse_args()
    if args.search and not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Please run 'pip install optuna'.")

    # load data (now returns train, val, test)
    train_matrix, val_matrix, test_matrix, n_users, n_items = load_data(
        dataset_name=args.dataset,
        max_reviews=args.subset_size,
        k_core=args.k_core,
        val_ratio=0.1
    )

    if args.search:
        print("\n--- INITIATING HYPERPARAMETER SEARCH MODE ---")
        search_space = {
            'n_clusters': [5, 10, 20],
            'gamma': [0.01, 0.1],
            'dropout': [0.3],
            'latent_dim': [32, 64, 128]
        }
        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        objective_with_args = partial(objective, args=args, train_matrix=train_matrix, val_matrix=val_matrix, test_matrix=test_matrix, n_users=n_users, n_items=n_items)
        n_trials = 1
        for param in search_space.values():
            n_trials *= len(param)
        print(f"Starting grid search with {n_trials} total trials.")
        study.optimize(objective_with_args, n_trials=n_trials)

        print("\n\n--- HYPERPARAMETER SEARCH COMPLETE ---")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best NDCG@10: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        results_list = []
        for trial in study.trials:
            params = trial.params
            all_metrics = trial.user_attrs.get("all_metrics", {})
            row = {**params, **all_metrics, 'value': trial.value, 'state': trial.state.name}
            results_list.append(row)
        df_results = pd.DataFrame(results_list)
        param_cols = list(study.best_params.keys())
        metric_cols = sorted([col for col in df_results.columns if col.startswith(('HR@', 'NDCG@'))])
        display_cols = ['value'] + param_cols + metric_cols + ['state']
        df_results = df_results[display_cols]
        print("\nFull Results Table (sorted by NDCG@10):")
        print(df_results.sort_values(by='value', ascending=False).to_string())
    else:
        print("\n--- INITIATING SINGLE RUN MODE ---")
        run_single_trial(args, train_matrix, val_matrix, test_matrix, n_users, n_items)


if __name__ == '__main__':
    main()
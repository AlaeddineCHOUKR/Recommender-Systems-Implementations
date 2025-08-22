# Implementations of Recommender System Algorithms

This repository contains PyTorch implementations of various collaborative filtering algorithms. Each algorithm is self-contained in its own directory.

## Implemented Algorithms
- **[DeepFuzzyCF](./DeepFuzzyCF):** An implementation of a Deep Embedded Fuzzy Clustering model for collaborative filtering.
- **[VAE-NCF](./VAE-NCF):** An implementation of VAEs to learn rich, latent representations of users and items, which are then fed into a Neural Collaborative Filtering (NCF) architecture to predict user-item interactions

## Features for DeepFuzzyCF
- Deep Autoencoder for learning user latent representations.
- Fuzzy C-Means style clustering objective integrated with reconstruction loss.
- Greedy layer-wise pretraining for the autoencoder.
- K-Means initialization of cluster centers in the latent space.
- Evaluation using standard recommendation metrics (HR@k, NDCG@k).
- Hyperparameter search powered by Optuna.
- Cluster visualization using t-SNE.



## Parameter List (DeepFuzzyCF)

The following command-line arguments can be used to configure the script (`AECLUST.py`).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Execution Mode** | | | |
| `--search` | flag | `False` | If present, enables hyperparameter search with Optuna instead of a single run. |
| **Data & Preprocessing** | | | |
| `--dataset` | string | `ml-100k` | Name of the dataset. Choices: `ml-100k`, `ml-1m`, `amazon-books`, `lastfm`. |
| `--subset_size` | int | `None` | Number of reviews to load for large datasets like Amazon Books. |
| `--k_core` | int | `5` | K-core filtering threshold. Removes users/items with fewer interactions. |
| **Model Architecture** | | | |
| `--clusters` | int | `15` | The number of user clusters to form. |
| `--latent_dim` | int | `64` | The dimensionality of the final latent space. Choices: `32`, `64`, `128`. |
| **Training & Optimization** | | | |
| `--finetune_epochs` | int | `50` | Maximum number of epochs for the main fine-tuning phase. |
| `--pretrain_layer_epochs` | int | `10` | Epochs per layer for the greedy layer-wise pretraining. |
| `--pretrain_global_epochs`| int | `10` | Epochs for the optional global pretraining after layer-wise. |
| `--batch_size` | int | `256` | Batch size for training and evaluation data loaders. |
| `--lr` | float | `1e-3` | Learning rate for pretraining. Fine-tuning LR is this value divided by 5. |
| `--patience` | int | `5` | Number of epochs with no improvement on validation loss to wait before early stopping. |
| **Regularization & Loss** | | | |
| `--dropout` | float | `0.5` | Dropout rate applied to the input layer during training. |
| `--alpha` | float | `1e-5` | Weight for the KL-divergence sparsity regularization loss. |
| `--rho` | float | `0.05` | The target activation value for the sparsity constraint. |
| `--gamma` | float | `0.01` | Weight for the clustering loss term during the fine-tuning phase. |
| **Evaluation & Miscellaneous**| | | |
| `--neighbors` | int | `50` | Number of neighbors to consider for Top-K recommendation during evaluation. |
| `--seed` | int | `42` | Random seed for ensuring reproducibility. |


## Features for DeepFuzzyCF
- Variational Autoencoders (VAEs) to learn user/item latent distributions.
- Neural Collaborative Filtering (NCF) core to model user-item interactions.
- Jointly optimized for both reconstruction and recommendation performance.
- Built for implicit feedback with a pointwise negative sampling strategy.
- Evaluates both ranking and prediction accuracy (HR/NDCG).

## Parameter List (VAE-NCF)

The following command-line arguments can be used to configure the script (`vae_ncf.py`).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Data & Preprocessing** | | | |
| `--dataset` | string | `ml-1m` | Name of the dataset. Choices: `ml-100k`, `ml-1m`, `amazon-books`, `lastfm`. |
| `--data_dir` | string | `../data` | Path to the parent data directory. |
| `--subset_size` | int | `None` | Number of reviews to load for large datasets (e.g., Amazon Books). |
| `--k_core` | int | `5` | K-core filtering threshold for users and items. |
| **Model Architecture** | | | |
| `--latent_dim` | int | `32` | The dimensionality of the latent space for both VAEs and GMF. |
| `--vae_dims` | string | `128,64` | Comma-separated hidden layer dimensions for the VAE encoder/decoder. |
| `--mlp_layers` | string | `64,32,16,8`| Comma-separated hidden layer dimensions for the NCF's MLP tower. |
| `--dropout` | float | `0.2` | Dropout rate for the MLP layers. |
| **Training & Optimization** | | | |
| `--epochs` | int | `50` | Maximum number of training epochs. |
| `--batch_size` | int | `256` | Batch size for training. |
| `--lr` | float | `1e-3` | Learning rate for the Adam optimizer. |
| `--num_negatives` | int | `4` | Number of negative items to sample for each positive interaction. |
| `--alpha` | float | `0.01` | Weighting factor for the VAE's reconstruction and KLD loss term. |
| **Evaluation & Miscellaneous**| | | |
| `--early_stop_patience`| int | `5` | Number of epochs with no improvement on validation NDCG@10 before stopping. |
| `--eval_batch_items` | int | `1024` | Number of items to process in a batch during ranking evaluation to manage memory. |
| `--seed` | int | `42` | Random seed for ensuring reproducibility. |

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/AlaeddineCHOURK-ALLAH/Recommender-Systems-Implementations.git
cd Recommender-Systems-Implementations
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows: .\venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
```

### 3. Install Dependencies
This will install all packages required for all algorithms in the repository.
```bash
pip install -r requirements.txt
```

## Usage

To run a specific algorithm, navigate into its directory first.

### Running DeepFuzzyCF

To run the model on the ML-100k dataset with default parameters:
```bash
cd DeepFuzzyCF
python AECLUST.py --dataset ml-100k
```

To specify different parameters, for example, for the Last.fm dataset:
```bash
cd DeepFuzzyCF
python AECLUST.py --dataset lastfm --clusters 20 --latent_dim 128 --neighbors 30 --finetune_epochs 100
```

To run a hyperparameter search for DeepFuzzyCF:
```bash
cd DeepFuzzyCF
python AECLUST.py --dataset ml-100k --search

```
### Running VAE-NCF
To train and evaluate the model on the ML-1M dataset with default parameters:
```bash
CD VAE-NCF
python vae_ncf.py --dataset ml-1m
```

To run with custom parameters, for example on the ML-100k dataset with a different latent dimension and learning rate:
```bash
CD VAE-NCF
python vae_ncf.py --dataset ml-100k --latent_dim 64 --lr 5e-4
```


To run another algorithm, you would first `cd ..` to go back to the root, and then `cd` into its directory.

## Citations

These implementations are inspired by the following papers:

> He, S., Li, T., Duan, Y., Yang, Z., & Li, F. (2019). VAE Based-NCF for Recommendation of Implicit Feedback. *2019 IEEE 8th Joint International Information Technology and Artificial Intelligence Conference (ITAIC)*, 512-516. https://ieeexplore.ieee.org/document/8785761

>Adel,B. (2022) Deep Embedded Fuzzy Clustering Model for Collaborative Filtering Recommender System Intelligent Automation & Soft Computing 2022, 33(1), 501-513. https://doi.org/10.32604/iasc.2022.022239

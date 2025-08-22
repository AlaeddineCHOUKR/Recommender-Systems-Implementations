# Implementations of Recommender System Algorithms

This repository contains PyTorch implementations of various collaborative filtering algorithms. Each algorithm is self-contained in its own directory.

## Implemented Algorithms
- **[DeepFuzzyCF](./DeepFuzzyCF):** An implementation of a Deep Embedded Fuzzy Clustering model for collaborative filtering.

## Features for DeepFuzzyCF
- Deep Autoencoder for learning user latent representations.
- Fuzzy C-Means style clustering objective integrated with reconstruction loss.
- Greedy layer-wise pretraining for the autoencoder.
- K-Means initialization of cluster centers in the latent space.
- Evaluation using standard recommendation metrics (HR@k, NDCG@k).
- Hyperparameter search powered by Optuna.
- Cluster visualization using t-SNE.

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

### 4. Download Datasets
Datasets are shared across all algorithms and must be placed in the top-level `data/` directory.

- **MovieLens 100K / 1M**: Download from the [GroupLens website](https://grouplens.org/datasets/movelens/). Unzip and place the `ml-100k` or `ml-1m` folder inside the `data/` directory.
- **Last.fm**: Download the "user_artists.dat" file from the [HetRec 2011 dataset](https://grouplens.org/datasets/hetrec-2011/). Create a `data/lastfm` folder and place the file inside.
- **Amazon Books**: Download the "Books" gzipped JSONL file from the [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html) and place it inside a `data/amazon-books` folder.

Your final `data` directory structure should be:
```
Recommender-Systems-Implementations/
└── data/
    ├── ml-100k/
    ├── ml-1m/
    ...
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
To run another algorithm, you would first `cd ..` to go back to the root, and then `cd` into its directory.
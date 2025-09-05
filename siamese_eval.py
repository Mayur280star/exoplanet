"""
exo_siamese_step8_eval.py

Step 8: Evaluation & Visualization
- Load trained Siamese FCNN
- Generate embeddings for all triplets
- Apply PCA / t-SNE for visualization
- Plot clusters of embeddings

Usage:
    python exo_siamese_step8_eval.py --triplets data/pairs/triplets.npz --model models/siamese_fcnn.pth
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# Dataset
# -----------------------------
class ExoplanetTripletDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.A = data["A"]
        self.P = data["P"]
        self.N = data["N"]

    def __len__(self):
        return self.A.shape[0]

    def __getitem__(self, idx):
        a = torch.tensor(self.A[idx], dtype=torch.float32)
        p = torch.tensor(self.P[idx], dtype=torch.float32)
        n = torch.tensor(self.N[idx], dtype=torch.float32)
        return a, p, n

# -----------------------------
# Model (same as step 7)
# -----------------------------
import torch.nn as nn

class SiameseFCNN(nn.Module):
    def __init__(self, input_len):
        super(SiameseFCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(16)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        return self.fc(self.feature_extractor(x))

    def forward(self, a, p, n):
        return self.forward_once(a), self.forward_once(p), self.forward_once(n)

# -----------------------------
# Generate Embeddings
# -----------------------------
def generate_embeddings(model, dataloader, device):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for a, p, n in dataloader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            ea, ep, en = model(a, p, n)
            embeddings.append(ea.cpu().numpy())
            embeddings.append(ep.cpu().numpy())
            embeddings.append(en.cpu().numpy())
            labels.extend(["anchor"] * len(ea))
            labels.extend(["positive"] * len(ep))
            labels.extend(["negative"] * len(en))
    embeddings = np.vstack(embeddings)
    return embeddings, np.array(labels)

# -----------------------------
# Visualization
# -----------------------------
def visualize_embeddings(embeddings, labels, method="pca"):
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for lbl, color in zip(["anchor", "positive", "negative"], ["blue", "green", "red"]):
        idxs = labels == lbl
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=lbl, alpha=0.6, s=20, c=color)
    plt.legend()
    plt.title(f"Siamese Embeddings ({method.upper()})")
    plt.grid(alpha=0.3)
    plt.show()

# -----------------------------
# CLI
# -----------------------------
def main(triplets, model_path, batch_size=64, device="cpu"):
    dataset = ExoplanetTripletDataset(triplets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_len = dataset.A.shape[2]
    model = SiameseFCNN(input_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    embeddings, labels = generate_embeddings(model, dataloader, device)

    print("✅ Generated embeddings:", embeddings.shape)

    visualize_embeddings(embeddings, labels, method="pca")
    visualize_embeddings(embeddings, labels, method="tsne")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate Siamese FCNN embeddings")
    p.add_argument("--triplets", default="data/pairs/triplets.npz")
    p.add_argument("--model", default="models/siamese_fcnn.pth")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    main(args.triplets, args.model, args.batch_size, args.device)

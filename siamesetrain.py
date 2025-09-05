"""
exo_siamese_step6-7.py

Steps:
6) Load triplets.npz into a PyTorch Dataset + DataLoader
7) Define Siamese FCNN with shared convolutional feature extractor
8) Train with Triplet Loss and save trained model

Usage:
    python exo_siamese_step6-7.py --triplets data/pairs/triplets.npz --epochs 20 --batch_size 64
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# -----------------------------
# Step 6: Dataset Loader
# -----------------------------
class ExoplanetTripletDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.A = data["A"]  # (N, 2, global_len)
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
# Step 7: Siamese FCNN
# -----------------------------
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
            nn.AdaptiveMaxPool1d(16)  # fixed-size output
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # embedding space
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

    def forward(self, a, p, n):
        return self.forward_once(a), self.forward_once(p), self.forward_once(n)

# -----------------------------
# Triplet Loss
# -----------------------------
def triplet_loss(a, p, n, margin=1.0):
    d_pos = torch.nn.functional.pairwise_distance(a, p)
    d_neg = torch.nn.functional.pairwise_distance(a, n)
    return torch.relu(d_pos - d_neg + margin).mean()

# -----------------------------
# Training Loop
# -----------------------------
def train(npz_path, epochs=10, batch_size=32, lr=1e-3, device="cpu", out_model="siamese_fcnn.pth"):
    dataset = ExoplanetTripletDataset(npz_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_len = dataset.A.shape[2]
    model = SiameseFCNN(input_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for a, p, n in dataloader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            embed_a, embed_p, embed_n = model(a, p, n)
            loss = triplet_loss(embed_a, embed_p, embed_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.4f}")

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_model)
    print(f"✅ Model saved at {out_model}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exoplanet Siamese FCNN training (steps 6-7)")
    p.add_argument("--triplets", default="data/pairs/triplets.npz", help="Path to triplets npz")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_model", default="models/siamese_fcnn.pth")
    args = p.parse_args()

    train(args.triplets, args.epochs, args.batch_size, args.lr, args.device, args.out_model)

# exo_siamese_step9_similarity.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from siamesetrain import SiameseFCNN  # ✅ make sure this points to your model file

# -----------------------------
# Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild model (same input length used in training)
input_len = 2001
model = SiameseFCNN(input_len).to(device)
model.load_state_dict(torch.load("models/siamese_fcnn.pth", map_location=device))
model.eval()

# -----------------------------
# Extractor wrapper
# -----------------------------
def get_single_embedding(x):
    """Run a single curve through the Siamese model encoder."""
    # ⚡ If your SiameseFCNN doesn't expose encoder, we assume encoder is defined inside.
    # Let's directly reuse its first sequential layers instead of forward()
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            encoder = module
            break
    else:
        raise RuntimeError("Could not find encoder inside SiameseFCNN")

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2, L)
    with torch.no_grad():
        emb = encoder(x)
    return emb.cpu().numpy().flatten()

# -----------------------------
# Load test triplets
# -----------------------------
data = np.load("data/pairs/triplets.npz")
A = data["A"]
P = data["P"]
N = data["N"]

print(f"Loaded test triplets: {A.shape[0]} examples")

# -----------------------------
# Compute embeddings
# -----------------------------
embeddings_A = np.array([get_single_embedding(a) for a in A])
embeddings_P = np.array([get_single_embedding(p) for p in P])
embeddings_N = np.array([get_single_embedding(n) for n in N])

print("✅ Embeddings generated")

# -----------------------------
# Distances & Similarities
# -----------------------------
def euclidean(a, b):
    return np.linalg.norm(a - b)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

dist_pos = [euclidean(a, p) for a, p in zip(embeddings_A, embeddings_P)]
dist_neg = [euclidean(a, n) for a, n in zip(embeddings_A, embeddings_N)]

sim_pos = [cosine(a, p) for a, p in zip(embeddings_A, embeddings_P)]
sim_neg = [cosine(a, n) for a, n in zip(embeddings_A, embeddings_N)]

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(dist_pos, bins=30, alpha=0.6, label="Anchor-Positive")
plt.hist(dist_neg, bins=30, alpha=0.6, label="Anchor-Negative")
plt.xlabel("Euclidean Distance")
plt.ylabel("Count")
plt.title("Distance Distribution")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(sim_pos, bins=30, alpha=0.6, label="Anchor-Positive")
plt.hist(sim_neg, bins=30, alpha=0.6, label="Anchor-Negative")
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.title("Similarity Distribution")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# ROC Curve (distance-based)
# -----------------------------
y_true = np.array([1] * len(dist_pos) + [0] * len(dist_neg))  # 1 = positive, 0 = negative
y_scores = np.array(dist_pos + dist_neg) * -1  # invert (smaller distance = more similar)

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Verification ROC")
plt.legend()
plt.show()

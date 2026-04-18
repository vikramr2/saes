import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

from sae import SparseAutoencoder, INPUT_DIM, DICT_SIZE, DEVICE

EMBEDDINGS_NPY = "../data/embeddings.npy"
CHECKPOINT_PATH = "../data/sae_checkpoint.pt"


def main():
    embeddings = np.load(EMBEDDINGS_NPY).astype(np.float32)
    x = torch.from_numpy(embeddings)
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=1, keepdim=True).mean()

    model = SparseAutoencoder(INPUT_DIM, DICT_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        recon, acts = model(x.to(DEVICE))

    acts = acts.cpu()
    recon = recon.cpu()

    # ── Sparsity ─────────────────────────────────────────────────────────────
    fired = (acts > 0)
    mean_active = fired.float().sum(dim=1).mean().item()
    dead_features = (fired.sum(dim=0) == 0).sum().item()
    print(f"Mean active features per sample : {mean_active:.1f} / {DICT_SIZE}  ({100*mean_active/DICT_SIZE:.2f}%)")
    print(f"Dead features (never fire)      : {dead_features} / {DICT_SIZE}  ({100*dead_features/DICT_SIZE:.1f}%)")

    # ── Reconstruction R² ────────────────────────────────────────────────────
    ss_res = (x - recon).pow(2).sum(dim=1)
    ss_tot = (x - x.mean(dim=0)).pow(2).sum(dim=1)
    r2 = (1 - ss_res / ss_tot).mean().item()
    print(f"Reconstruction R²               : {r2:.4f}  (1.0 = perfect)")

    # ── Activation frequency histogram ───────────────────────────────────────
    freq = fired.float().mean(dim=0).numpy()
    buckets = [0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.01]
    labels  = ["=0 (dead)", "<0.01%", "<0.1%", "<1%", "<10%", "<50%", "≥50%"]
    counts  = Counter()
    for f in freq:
        for i, b in enumerate(buckets[1:]):
            if f < b:
                counts[labels[i]] += 1
                break
    print("\nFeature activation frequency distribution:")
    for label in labels:
        bar = "█" * (counts[label] * 40 // max(counts.values(), default=1))
        print(f"  {label:>12s}  {counts[label]:5d}  {bar}")

    # ── Top alive features ───────────────────────────────────────────────────
    alive = [(i, float(freq[i])) for i in range(DICT_SIZE) if freq[i] > 0]
    alive.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 most-firing features:")
    for feat, f in alive[:10]:
        print(f"  feature {feat:5d}  fires on {100*f:.2f}% of samples")
    print(f"\nBottom 10 alive features:")
    for feat, f in alive[-10:]:
        print(f"  feature {feat:5d}  fires on {100*f:.4f}% of samples")


if __name__ == "__main__":
    main()

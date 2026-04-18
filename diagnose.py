import argparse
import numpy as np
import torch
from collections import Counter

from sae_loader import load_sae_acts, EMBEDDINGS_NPY, DEVICE, _normalize

DATA_DIR = "data"
CHECKPOINT_PATH = f"{DATA_DIR}/sota_sae_checkpoint.pt"


def main(sae_type: str, checkpoint: str = None):
    print(f"Loading SAE activations ({sae_type}{f' @ {checkpoint}' if checkpoint else ''})...")
    acts, dict_size = load_sae_acts(sae_type, checkpoint)

    raw = np.load(EMBEDDINGS_NPY).astype(np.float32)
    x = _normalize(torch.from_numpy(raw))

    import sys
    sys.path.insert(0, "model")
    from sae import SparseAutoencoder, INPUT_DIM, DICT_SIZE
    if sae_type == "custom":
        from sae_loader import CUSTOM_CHECKPOINT, _load_custom
        model, _ = _load_custom(CUSTOM_CHECKPOINT)
        with torch.no_grad():
            recon, _ = model(x.to(DEVICE))[:2]
        recon = recon.cpu()
    else:
        recon = None

    acts_t = torch.from_numpy(acts)
    fired = acts_t > 0
    mean_active = fired.float().sum(dim=1).mean().item()
    dead_features = (fired.sum(dim=0) == 0).sum().item()
    print(f"Mean active features per sample : {mean_active:.1f} / {dict_size}  ({100*mean_active/dict_size:.2f}%)")
    print(f"Dead features (never fire)      : {dead_features} / {dict_size}  ({100*dead_features/dict_size:.1f}%)")

    if recon is not None:
        ss_res = (x - recon).pow(2).sum(dim=1)
        ss_tot = (x - x.mean(dim=0)).pow(2).sum(dim=1)
        r2 = (1 - ss_res / ss_tot).mean().item()
        print(f"Reconstruction R²               : {r2:.4f}  (1.0 = perfect)")

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

    alive = [(i, float(freq[i])) for i in range(dict_size) if freq[i] > 0]
    alive.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 most-firing features:")
    for feat, f in alive[:10]:
        print(f"  feature {feat:5d}  fires on {100*f:.2f}% of samples")
    print(f"\nBottom 10 alive features:")
    for feat, f in alive[-10:]:
        print(f"  feature {feat:5d}  fires on {100*f:.4f}% of samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae", choices=["custom", "saelens"], default="custom")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args()
    main(args.sae, args.checkpoint)

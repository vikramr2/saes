"""
Common interface for loading any SAE and computing activations.

Usage:
    acts, dict_size = load_sae_acts(sae_type="custom")   # our trained SAE
    acts, dict_size = load_sae_acts(sae_type="saelens")  # SAELens TopK SAE
"""
import sys
import numpy as np
import torch

DATA_DIR = "data"
EMBEDDINGS_NPY = f"{DATA_DIR}/embeddings.npy"
EMBEDDING_IDS_NPY = f"{DATA_DIR}/embedding_ids.npy"
CUSTOM_CHECKPOINT = f"{DATA_DIR}/sae_checkpoint.pt"
SAELENS_CHECKPOINT = f"{DATA_DIR}/saelens_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=1, keepdim=True).mean()
    return x


def _load_custom(checkpoint_path: str):
    sys.path.insert(0, "model")
    from sae import SparseAutoencoder, INPUT_DIM, DICT_SIZE
    model = SparseAutoencoder(INPUT_DIM, DICT_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model, DICT_SIZE


def _load_saelens(checkpoint_path: str):
    sys.path.insert(0, "model")
    from saelens import TopKSAE
    state = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = state["cfg"]
    model = TopKSAE(cfg["d_in"], cfg["d_sae"], cfg["k"]).to(DEVICE)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, cfg["d_sae"]


def load_sae_acts(sae_type: str = "custom", checkpoint_path: str = None) -> tuple[np.ndarray, int]:
    raw = np.load(EMBEDDINGS_NPY).astype(np.float32)
    x = _normalize(torch.from_numpy(raw)).to(DEVICE)

    if sae_type == "custom":
        ckpt = checkpoint_path or CUSTOM_CHECKPOINT
        model, dict_size = _load_custom(ckpt)
        with torch.no_grad():
            _, acts = model(x)[:2]
    elif sae_type == "saelens":
        ckpt = checkpoint_path or SAELENS_CHECKPOINT
        model, dict_size = _load_saelens(ckpt)
        with torch.no_grad():
            acts, _ = model.encode(x)
    else:
        raise ValueError(f"Unknown sae_type '{sae_type}'. Choose 'custom' or 'saelens'.")

    return acts.cpu().numpy(), dict_size

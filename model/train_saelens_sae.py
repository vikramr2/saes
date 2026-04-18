"""
Train a SOTA TopK SAE on our SPECTER2 embeddings.
Saves data/sota_sae_checkpoint.pt compatible with sae_loader.py.
"""
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

EMBEDDINGS_NPY = "../data/embeddings.npy"
OUTPUT_PATH = "../data/sota_sae_checkpoint.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

INPUT_DIM = 768
DICT_SIZE = 2048
TOP_K = 64
BATCH_SIZE = 256
EPOCHS = 100
LR = 2e-4
AUX_COEFF = 1.0 / TOP_K
DEAD_THRESHOLD = 1e-4
EMA_DECAY = 0.99


def normalize(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=1, keepdim=True).mean()
    return x


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, dict_size: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.encoder(x)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return acts, pre

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts)

    def forward(self, x: torch.Tensor):
        acts, pre = self.encode(x)
        return self.decode(acts), acts, pre

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)


def train():
    print(f"Device: {DEVICE}")
    raw = np.load(EMBEDDINGS_NPY).astype(np.float32)
    x = normalize(torch.from_numpy(raw)).to(DEVICE)
    print(f"Embeddings: {x.shape}  →  dict_size={DICT_SIZE}  k={TOP_K}")

    model = TopKSAE(INPUT_DIM, DICT_SIZE, TOP_K).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    feat_ema = torch.zeros(DICT_SIZE, device=DEVICE)

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(x), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        total_loss = mean_active = 0.0
        model.train()

        for (batch,) in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            recon, acts, pre = model(batch)

            with torch.no_grad():
                feat_ema = EMA_DECAY * feat_ema + (1 - EMA_DECAY) * (acts > 0).float().mean(dim=0)

            recon_loss = (batch - recon).pow(2).sum(dim=-1).mean()

            # Aux loss on dead features
            dead = feat_ema < DEAD_THRESHOLD
            aux_loss = torch.tensor(0.0, device=DEVICE)
            if dead.any():
                residual = (batch - recon).detach()
                dead_pre = pre[:, dead]
                dead_acts = torch.zeros_like(dead_pre)
                k = min(int(dead.sum().item()), TOP_K)
                tv, ti = dead_pre.topk(k, dim=-1)
                dead_acts.scatter_(-1, ti, torch.relu(tv))
                recon_dead = dead_acts @ model.decoder.weight[:, dead].T
                aux_loss = (residual - recon_dead).pow(2).sum(dim=-1).mean()

            loss = recon_loss + AUX_COEFF * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            total_loss += loss.item()
            mean_active += (acts > 0).float().mean().item()

        n = len(loader)
        n_dead = int((feat_ema < DEAD_THRESHOLD).sum())
        print(f"Epoch {epoch:3d} | loss {total_loss/n:.4f} | "
              f"active {mean_active/n*DICT_SIZE:.1f}/{DICT_SIZE} | dead {n_dead}")

    torch.save({
        "state_dict": model.state_dict(),
        "cfg": {"d_in": INPUT_DIM, "d_sae": DICT_SIZE, "k": TOP_K},
    }, OUTPUT_PATH)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    train()

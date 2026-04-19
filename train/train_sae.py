import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, "..")
from model.sae import SparseAutoencoder, INPUT_DIM, DICT_SIZE, DEVICE

EMBEDDINGS_NPY = "../data/embeddings.npy"
CHECKPOINT_PATH = "../data/sae_checkpoint.pt"

L1_COEFF = 2e-2
AUX_COEFF = 1.0 / 4
DEAD_THRESHOLD = 1e-4
EMA_DECAY = 0.99
LR = 1e-4
BATCH_SIZE = 256
EPOCHS = 120


def normalize(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=1, keepdim=True).mean()
    return x


def train():
    print(f"Device: {DEVICE}")
    raw = np.load(EMBEDDINGS_NPY).astype(np.float32)
    x = normalize(torch.from_numpy(raw)).to(DEVICE)

    loader = DataLoader(TensorDataset(x), batch_size=BATCH_SIZE, shuffle=True)
    model = SparseAutoencoder(INPUT_DIM, DICT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    feat_ema = torch.ones(DICT_SIZE, device=DEVICE) * 0.5

    for epoch in range(1, EPOCHS + 1):
        total_loss = recon_total = sparsity_total = mean_active = 0.0
        model.train()

        for (batch,) in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            recon, acts = model(batch)

            with torch.no_grad():
                feat_ema = EMA_DECAY * feat_ema + (1 - EMA_DECAY) * (acts > 0).float().mean(dim=0)

            recon_loss = (batch - recon).pow(2).sum(dim=-1).mean()
            sparsity_loss = acts.abs().sum(dim=-1).mean()

            dead = feat_ema < DEAD_THRESHOLD
            aux_loss = torch.tensor(0.0, device=DEVICE)
            if dead.any():
                residual = (batch - recon).detach()
                # encode residual through dead features only
                with torch.no_grad():
                    dead_pre = model.encoder(residual)[:, dead]
                dead_acts = torch.zeros_like(dead_pre)
                k = min(int(dead.sum().item()), 32)
                tv, ti = dead_pre.topk(k, dim=-1)
                dead_acts.scatter_(-1, ti, torch.relu(tv))
                recon_dead = dead_acts @ model.decoder.weight[:, dead].T
                aux_loss = (residual - recon_dead).pow(2).sum(dim=-1).mean()

            loss = recon_loss + L1_COEFF * sparsity_loss + AUX_COEFF * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            total_loss += loss.item()
            recon_total += recon_loss.item()
            sparsity_total += sparsity_loss.item()
            mean_active += (acts > 0).float().mean().item()

        n = len(loader)
        n_dead = int((feat_ema < DEAD_THRESHOLD).sum())
        print(
            f"Epoch {epoch:3d} | loss {total_loss/n:.4f} | "
            f"recon {recon_total/n:.4f} | sparsity {sparsity_total/n:.4f} | "
            f"active {mean_active/n*DICT_SIZE:.1f}/{DICT_SIZE} | dead {n_dead}"
        )

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved → {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()

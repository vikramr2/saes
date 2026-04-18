import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

EMBEDDINGS_NPY = "../data/embeddings.npy"
CHECKPOINT_PATH = "../data/sae_checkpoint.pt"

# SAE hyperparameters
INPUT_DIM = 768
EXPANSION = 10          # dictionary size = INPUT_DIM * EXPANSION
DICT_SIZE = INPUT_DIM * EXPANSION
L1_COEFF = 2e-2        # sparsity penalty weight
LR = 1e-4
BATCH_SIZE = 256
EPOCHS = 120
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dict_size: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        # Normalise decoder columns to unit norm at init
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def forward(self, x):
        pre_acts = self.encoder(x)
        acts = torch.relu(pre_acts)           # sparse feature activations
        recon = self.decoder(acts)
        return recon, acts

    @torch.no_grad()
    def normalize_decoder(self):
        """Keep decoder columns unit-norm (prevents trivial shrinking solution)."""
        self.decoder.weight.data = nn.functional.normalize(
            self.decoder.weight.data, dim=0
        )


def sae_loss(x, recon, acts, l1_coeff):
    recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
    sparsity_loss = acts.abs().sum(dim=-1).mean()
    return recon_loss + l1_coeff * sparsity_loss, recon_loss, sparsity_loss


def train():
    print(f"Device: {DEVICE}")

    embeddings = np.load(EMBEDDINGS_NPY).astype(np.float32)
    x = torch.from_numpy(embeddings)
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=1, keepdim=True).mean()  # scale to unit mean norm

    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SparseAutoencoder(INPUT_DIM, DICT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        total_loss = recon_total = sparsity_total = 0.0
        mean_active = 0.0
        model.train()

        for (batch,) in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            batch = batch.to(DEVICE)
            recon, acts = model(batch)
            loss, recon_loss, sparsity_loss = sae_loss(batch, recon, acts, L1_COEFF)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            total_loss += loss.item()
            recon_total += recon_loss.item()
            sparsity_total += sparsity_loss.item()
            mean_active += (acts > 0).float().mean().item()

        n = len(loader)
        avg_active = mean_active / n * DICT_SIZE
        target = "✓" if avg_active <= 50 else f"target ≤50"
        print(
            f"Epoch {epoch:3d} | loss {total_loss/n:.4f} | "
            f"recon {recon_total/n:.4f} | sparsity {sparsity_total/n:.4f} | "
            f"mean active {avg_active:.1f}/{DICT_SIZE} {target}"
        )

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved checkpoint → {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()

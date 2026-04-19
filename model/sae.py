import torch
import torch.nn as nn

INPUT_DIM = 768
EXPANSION = 10
DICT_SIZE = INPUT_DIM * EXPANSION
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dict_size: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def forward(self, x):
        pre_acts = self.encoder(x)
        acts = torch.relu(pre_acts)
        recon = self.decoder(acts)
        return recon, acts

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = nn.functional.normalize(
            self.decoder.weight.data, dim=0
        )

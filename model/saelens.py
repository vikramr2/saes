import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


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

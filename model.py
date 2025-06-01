import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm

class SDFNet(nn.Module):
    def __init__(self,
                 pe_freqs: int = 6,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 skip_connection_at: int = 4):
        super().__init__()
        self.pe_freqs = pe_freqs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip = skip_connection_at

        # PE(xyz): 3 + 2*pe_freqs*3
        self.pe_dim = 3 + 2 * pe_freqs * 3
        self.input_dim = self.pe_dim
        self.pe_mask = torch.ones(self.pe_freqs, dtype=torch.bool)

        layers = []
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            if i == self.skip:
                in_dim += self.input_dim
            lin = weight_norm(nn.Linear(in_dim, hidden_dim))
            layers += [lin, nn.Softplus(beta=100)]
        self.layers = nn.ModuleList(layers)

        self.sdf_out     = nn.Linear(hidden_dim, 1)  # SDF prediction
        self.rgb_out     = nn.Linear(hidden_dim, 3)  # RGB prediction
        self.feature_out = nn.Linear(hidden_dim, hidden_dim)  # optional feature head

    def pos_enc(self, x: torch.Tensor) -> torch.Tensor:
        enc = [x]
        for i in range(self.pe_freqs):
            if self.pe_mask[i]:
                freq = (2.0 ** i) * math.pi
                enc.append(torch.sin(freq * x))
                enc.append(torch.cos(freq * x))
            else:
                enc.append(torch.zeros_like(x))
                enc.append(torch.zeros_like(x))
        return torch.cat(enc, dim=-1)

    def forward(self, x: torch.Tensor):
        """
        Args:
          x: [N,3] = xyz
        Returns:
          sdf: [N]
          rgb_pred: [N, 3]
        """
        pe = self.pos_enc(x)  # [N, pe_dim]
        net_input = pe

        h = net_input
        for i in range(self.num_layers):
            if i == self.skip:
                h = torch.cat([h, net_input], dim=-1)
            h = self.layers[2 * i + 1](self.layers[2 * i](h))

        sdf = self.sdf_out(h).squeeze(-1)         # [N]
        rgb_pred = torch.sigmoid(self.rgb_out(h)) # [N, 3]

        return sdf, rgb_pred

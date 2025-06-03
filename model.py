import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm

class SDFNet(nn.Module):
    def __init__(self,
                 pe_freqs: int = 6,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 skip_connection_at: int = 4,
                 use_viewdirs: bool = True):
        super().__init__()
        self.pe_freqs = pe_freqs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip = skip_connection_at
        self.use_viewdirs = use_viewdirs

        self.pe_dim = 3 + 2 * pe_freqs * 3  # positional encoding for xyz
        self.input_dim = self.pe_dim

        self.pe_mask = torch.ones(self.pe_freqs, dtype=torch.bool)

        # Geometry backbone (shared)
        layers = []
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            if i == self.skip:
                in_dim += self.input_dim
            layers += [weight_norm(nn.Linear(in_dim, hidden_dim)),
                       nn.Softplus(beta=100)]
        self.layers = nn.ModuleList(layers)

        self.sdf_out = nn.Linear(hidden_dim, 1)

        # Optional view direction input for RGB
        view_dim = 3 if use_viewdirs else 0
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + view_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  # output RGB in [0,1]
        )

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

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor = None, return_rgb=True):
        """
        Args:
            x: [N, 3] → xyz positions
            view_dirs: [N, 3], optional viewing directions
            return_rgb: whether to compute rgb output
        Returns:
            sdf: [N]
            rgb_pred: [N, 3] (or None if return_rgb=False)
        """
        pe = self.pos_enc(x)
        h = pe
        for i in range(self.num_layers):
            if i == self.skip:
                h = torch.cat([h, pe], dim=-1)
            h = self.layers[2 * i + 1](self.layers[2 * i](h))

        sdf = self.sdf_out(h).squeeze(-1)  # [N]

        if not return_rgb:
            return sdf, None

        # compute RGB only if required
        if self.use_viewdirs:
            assert view_dirs is not None, "view_dirs must be provided if use_viewdirs=True and return_rgb=True"
            h_rgb = torch.cat([h, view_dirs], dim=-1)
        else:
            h_rgb = h

        rgb_pred = self.rgb_head(h_rgb)  # [N, 3]
        return sdf, rgb_pred
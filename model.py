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
                 use_viewdirs: bool = True,
                 view_pe_freqs: int = 4,
                 use_sigmoid_rgb: bool = False):
        super().__init__()
        self.pe_freqs = pe_freqs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip = skip_connection_at
        self.use_viewdirs = use_viewdirs
        self.view_pe_freqs = view_pe_freqs
        self.use_sigmoid_rgb = use_sigmoid_rgb

        self.pe_mask = torch.ones(self.pe_freqs, dtype=torch.bool)

        self.pe_dim = 3 + 2 * pe_freqs * 3  # for xyz
        self.input_dim = self.pe_dim

        # === Geometry backbone ===
        layers = []
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            if i == self.skip:
                in_dim += self.input_dim
            layers += [
                weight_norm(nn.Linear(in_dim, hidden_dim)),
                nn.Softplus(beta=100)
            ]
        self.layers = nn.ModuleList(layers)
        self.sdf_out = nn.Linear(hidden_dim, 1)

        # === RGB head ===
        view_dim = 0
        if use_viewdirs:
            view_dim = 3 + 3 * 2 * view_pe_freqs  # original + sin/cos encoded

        self.rgb_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + view_dim),
            nn.Linear(hidden_dim + view_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid() if use_sigmoid_rgb else nn.Identity()
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

    def view_enc(self, d: torch.Tensor) -> torch.Tensor:
        """Encode viewing directions (if enabled)"""
        if self.view_pe_freqs == 0:
            return d
        out = [d]
        for i in range(self.view_pe_freqs):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * d))
            out.append(torch.cos(freq * d))
        return torch.cat(out, dim=-1)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor = None, return_rgb=True):
        """
        Args:
            x: [N, 3]
            view_dirs: [N, 3]
        Returns:
            sdf: [N], rgb_pred: [N, 3] (or None)
        """
        pe = self.pos_enc(x)
        h = pe
        for i in range(self.num_layers):
            if i == self.skip:
                h = torch.cat([h, pe], dim=-1)
            h = self.layers[2 * i + 1](self.layers[2 * i](h))

        sdf = self.sdf_out(h).squeeze(-1)

        if not return_rgb:
            return sdf, None

        # === RGB forward path ===
        if self.use_viewdirs:
            assert view_dirs is not None
            view_enc = self.view_enc(view_dirs)
            h_rgb = torch.cat([h, view_enc], dim=-1)
        else:
            h_rgb = h

        rgb_pred = self.rgb_head(h_rgb)  # sigmoid (0-1) or linear
        return sdf, rgb_pred

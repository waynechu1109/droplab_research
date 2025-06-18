import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm

class SDFNet(nn.Module):
    def __init__(self, pe_freqs, hidden_dim=64, num_layers=8, skip_connection_at=4):
        super().__init__()
        self.pe_freqs = pe_freqs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip = skip_connection_at

        self.register_buffer("pe_mask", torch.ones(self.pe_freqs, dtype=torch.bool))
        # Only PE for xyz (3D)
        self.pe_dim = 3 + 2 * pe_freqs * 3  # xyz + 2*freqs*xyz
        self.input_dim = self.pe_dim + 3    # xyz PE + rgb

        # Main MLP
        layers = []
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            if i == self.skip:
                in_dim += self.input_dim
            layers.append(weight_norm(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.Softplus(beta=100))
        self.layers = nn.ModuleList(layers)
        self.sdf_out = nn.Linear(hidden_dim, 1)

    def pos_enc(self, xyz):
        # xyz: [N, 3]
        enc = [xyz]
        for i in range(self.pe_freqs):
            if self.pe_mask[i]:
                freq = (2.0 ** i) * math.pi
                enc.append(torch.sin(freq * xyz))
                enc.append(torch.cos(freq * xyz))
            else:
                enc.append(torch.zeros_like(xyz))
                enc.append(torch.zeros_like(xyz))
        return torch.cat(enc, dim=-1)

    def forward(self, x):
        # x: [N, 6], (x, y, z, r, g, b)
        xyz = x[:, :3]
        rgb = x[:, 3:]
        pe_xyz = self.pos_enc(xyz)
        h = torch.cat([pe_xyz, rgb], dim=-1)
        for i in range(self.num_layers):
            if i == self.skip:
                h = torch.cat([h, pe_xyz, rgb], dim=-1)
            h = self.layers[2 * i + 1](self.layers[2 * i](h))
        sdf6d = self.sdf_out(h).squeeze(-1)
        return sdf6d


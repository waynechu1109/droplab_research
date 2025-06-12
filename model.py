import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm

class SDFNet(nn.Module):
    def __init__(self,
                 pe_freqs,
                 hidden_dim=256,
                 num_layers=8,
                 skip_connection_at=4,
                ):
        super().__init__()
        self.pe_freqs = pe_freqs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip = skip_connection_at

        self.register_buffer("pe_mask", torch.ones(self.pe_freqs, dtype=torch.bool))
        # PE applied to xyzrgb (6D)
        self.pe_dim = 6 + 2 * pe_freqs * 6
        self.input_dim = self.pe_dim

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

    def pos_enc(self, x):
        # x: [N, 6]
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

    def gradient(self, x):
        # For Eikonal loss etc.
        x.requires_grad_(True)
        sdf = self.forward(x)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return gradients.unsqueeze(1)  # [N, 1, 6]
    
    def forward(self, x):
        # x: [N, 6], (x, y, z, r, g, b)
        pe = self.pos_enc(x)
        h = pe
        for i in range(self.num_layers):
            if i == self.skip:
                h = torch.cat([h, pe], dim=-1)
            h = self.layers[2 * i + 1](self.layers[2 * i](h))
        sdf6d = self.sdf_out(h).squeeze(-1) # [N]
        return sdf6d

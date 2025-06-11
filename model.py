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
                 use_viewdirs=True,
                 view_pe_freqs=6,
                 use_tanh_rgb=True,
                 use_feature_vector=True,
                 feature_vector_size=64):
        super().__init__()
        self.pe_freqs = pe_freqs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip = skip_connection_at
        self.use_viewdirs = use_viewdirs
        self.view_pe_freqs = view_pe_freqs
        self.use_tanh_rgb = use_tanh_rgb
        self.use_feature_vector = use_feature_vector
        self.feature_vector_size = feature_vector_size

        self.register_buffer("pe_mask", torch.ones(self.pe_freqs, dtype=torch.bool))
        self.pe_dim = 3 + 2 * pe_freqs * 3  # PE applied to xyz
        self.input_dim = self.pe_dim
        self.rgb_input_dim = 0

        # Geometry branch
        layers = []
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            if i == self.skip:
                in_dim += self.input_dim
            layers.append(weight_norm(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.Softplus(beta=100))
        self.layers = nn.ModuleList(layers)

        self.sdf_out = nn.Linear(hidden_dim, 1)

        # Optional feature vector branch
        if use_feature_vector:
            self.feature_out = nn.Linear(hidden_dim, feature_vector_size)

        # RGB branch
        self.rgb_input_dim = 6 + feature_vector_size + hidden_dim + (3 if not use_viewdirs else (3 + 2 * view_pe_freqs * 3))

        self.rgb_head = nn.Sequential(
            nn.LayerNorm(self.rgb_input_dim),
            nn.Linear(self.rgb_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Tanh() if use_tanh_rgb else nn.Sigmoid()
        )

    def gradient(self, x):
        x.requires_grad_(True)
        sdf, _ = self.forward(x, return_rgb=False)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return gradients.unsqueeze(1)  # [N, 1, 3]

    def pos_enc(self, x):
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

    def view_enc(self, d):
        if self.view_pe_freqs == 0:
            return d
        out = [d]
        for i in range(self.view_pe_freqs):
            freq = (2.0 ** i) * math.pi
            out.append(torch.sin(freq * d))
            out.append(torch.cos(freq * d))
        return torch.cat(out, dim=-1)

    def forward(self, x, view_dirs=None, return_rgb=True, detach_sdf=False, compute_grad=False):
        pe = self.pos_enc(x)
        h = pe
        for i in range(self.num_layers):
            if i == self.skip:
                h = torch.cat([h, pe], dim=-1)
            h = self.layers[2 * i + 1](self.layers[2 * i](h))

        sdf = self.sdf_out(h).squeeze(-1)

        if not return_rgb:
            return sdf, None

        # Prevent RGB head from backpropagating into SDF branch
        h_rgb = h.detach() if detach_sdf else h

        if self.use_feature_vector:
            feat = self.feature_out(h_rgb)
        else:
            feat = h_rgb

        normals = torch.zeros_like(x)  # safe default
        if self.use_viewdirs:
            assert view_dirs is not None, "view_dirs required for RGB prediction"
            view_pe = self.view_enc(view_dirs)
            if compute_grad:
                x.requires_grad_(True)
                normals = self.gradient(x)[:, 0, :].detach()
            rgb_input = torch.cat([x, normals, feat, view_pe, h_rgb], dim=-1)
        else:
            rgb_input = torch.cat([x, normals, feat, h_rgb], dim=-1)

        rgb = self.rgb_head(rgb_input)
        if self.use_tanh_rgb:
            rgb = (rgb + 1.0) / 2.0  # Rescale tanh output to [0, 1]

        return sdf, rgb

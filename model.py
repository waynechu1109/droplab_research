import torch
import torch.nn as nn
# import tinycudann as tcnn  # Instant-NGP 的 CUDA 加速 hash encoding 實現
import math
from torch.nn.utils import weight_norm

# model architecture
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

        # PE for xyz, so input_dim = 3 + 2*pe_freqs*3
        self.pe_dim = 3 + 2 * pe_freqs * 3
        self.input_dim = self.pe_dim + 3  # + RGB

        layers = []
        for i in range(num_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            if i == self.skip:
                in_dim += self.input_dim
            lin = weight_norm(nn.Linear(in_dim, hidden_dim))
            layers += [lin, nn.Softplus(beta=100)]
        self.layers = nn.ModuleList(layers)

        # 输出 signed distance (1 维) + geometry feature (hidden_dim 维，暂不返回)
        self.sdf_out     = nn.Linear(hidden_dim, 1)
        self.feature_out = nn.Linear(hidden_dim, hidden_dim)  # 保留以备后续使用

    def pos_enc(self, x: torch.Tensor) -> torch.Tensor:
        enc = [x]
        for i in range(self.pe_freqs):
            freq = (2.0 ** i) * math.pi
            enc.append(torch.sin(freq * x))
            enc.append(torch.cos(freq * x))
        return torch.cat(enc, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [N,6] = [xyz, rgb]
        Returns:
          d: [N] predicted signed distance
        """
        # 拆分位置和颜色（后者暂不参与 SDF 计算）
        p = x[:, :3]  # [N,3]
        rgb = x[:,3:]

        # 对 xyz 做 PE 并通过 MLP
        pe = self.pos_enc(p)  # [N, input_dim]
        input_with_color = torch.cat([pe, rgb], dim=-1)  # 加上顏色資訊

        h = input_with_color
        for i in range(self.num_layers):
            if i == self.skip:
                h = torch.cat([h, input_with_color], dim=-1)
            lin = self.layers[2*i]
            act = self.layers[2*i + 1]
            h = act(lin(h))

        d = self.sdf_out(h).squeeze(-1)  # [N]
        # geometry feature g = self.feature_out(h)  # [N, hidden_dim], 暂不返回

        return d
    




# class SDFNet(nn.Module):
#     def __init__(self, input_dim=6, hidden_dim=128, num_layers=8, skip_connection_at=4, pe_freqs=6):
#         super().__init__()
#         self.num_layers = num_layers
#         self.skip_connection_at = skip_connection_at
#         self.pe_freqs = pe_freqs

#         self.input_dim = 3 + 2 * pe_freqs * 3 + 3  # PE(xyz) + RGB = final input dim
#         layers = []
#         for i in range(num_layers):
#             if i == 0:
#                 in_dim = self.input_dim
#             elif i == skip_connection_at:
#                 in_dim = hidden_dim + self.input_dim  # skip connection
#             else:
#                 in_dim = hidden_dim

#             layers.append(nn.Linear(in_dim, hidden_dim))
#             layers.append(nn.Softplus(beta=800))
#             # layers.append(nn.ReLU())
#         self.hidden_layers = nn.ModuleList(layers)
#         self.output_layer = nn.Linear(hidden_dim, 1)

#     # position encoding: 透過多個不同頻率的 sin/cos，把座標映射到更豐富的頻域空間，有助捕捉細節。
#     def pos_enc(self, x):
#         enc = [x]
#         for i in range(self.pe_freqs):
#             enc.append(torch.sin((2 ** i) * math.pi * x))
#             enc.append(torch.cos((2 ** i) * math.pi * x))
#         return torch.cat(enc, dim=-1)

#     def forward(self, x):  # x: [N, 6] = [xyz, rgb]
#         pos = x[:, :3]                  # [N, 3]
#         color = x[:, 3:]                # [N, 3]
#         pos_encoded = self.pos_enc(pos)  # [N, 3 + 2*pe_freqs*3]
#         net_input = torch.cat([pos_encoded, color], dim=-1)  # [N, input_dim]

#         h = net_input
#         for i in range(self.num_layers):
#             lin = self.hidden_layers[i * 2]
#             act = self.hidden_layers[i * 2 + 1]

#             # skip connection: 讓後續層直接存取最初的特徵，減輕梯度消失，並加速學習。
#             if i == self.skip_connection_at:
#                 h = torch.cat([h, net_input], dim=-1)

#             h = act(lin(h))

#         return self.output_layer(h).squeeze(-1)
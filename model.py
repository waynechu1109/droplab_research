import torch
import torch.nn as nn
import tinycudann as tcnn  # Instant-NGP 的 CUDA 加速 hash encoding


class SDFNet(nn.Module):
    def __init__(self,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 hash_config: dict = {
                    "otype": "HashGrid",
                    "n_levels": 20,
                    "n_features_per_level": 2,
                    "base_resolution": 32,
                    "per_level_scale": 1.5,
                    "log2_hashmap_size": 21
                    }
                ):
        super().__init__()

        self.input_dim = 3  # xyz only
        self.rgb_dim = 3    # rgb
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 使用 tinycudann 的 hash grid encoder
        self.encoder = tcnn.Encoding(n_input_dims=3, encoding_config=hash_config)

        mlp_layers = []
        in_dim = self.encoder.n_output_dims + self.rgb_dim  # 加上 RGB 資訊
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1  # 最後一層輸出 SDF
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                mlp_layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N,6] = [xyz, rgb]
        Returns:
            d: [N] predicted signed distance
        """
        p = x[:, :3]   # [N, 3]
        rgb = x[:, 3:] # [N, 3]
        encoded = self.encoder(p)  # [N, C]
        h = torch.cat([encoded, rgb], dim=-1)  # [N, C+3]
        sdf = self.mlp(h).squeeze(-1)
        return sdf


# import torch
# import torch.nn as nn
# # import tinycudann as tcnn  # Instant-NGP 的 CUDA 加速 hash encoding 實現
# import math
# from torch.nn.utils import weight_norm

# # model architecture
# class SDFNet(nn.Module):
#     def __init__(self,
#                  pe_freqs: int = 6,
#                  hidden_dim: int = 256,
#                  num_layers: int = 8,
#                  skip_connection_at: int = 4):
#         super().__init__()
#         self.pe_freqs = pe_freqs
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.skip = skip_connection_at

#         # PE for xyz, so input_dim = 3 + 2*pe_freqs*3
#         self.pe_dim = 3 + 2 * pe_freqs * 3
#         self.input_dim = self.pe_dim + 3  # + RGB
#         self.pe_mask = torch.ones(self.pe_freqs, dtype=torch.bool)  # default: all enabled

#         layers = []
#         for i in range(num_layers):
#             in_dim = self.input_dim if i == 0 else hidden_dim
#             if i == self.skip:
#                 in_dim += self.input_dim
#             lin = weight_norm(nn.Linear(in_dim, hidden_dim))
#             layers += [lin, nn.Softplus(beta=100)]
#         self.layers = nn.ModuleList(layers)

#         # 输出 signed distance (1 维) + geometry feature (hidden_dim 维，暂不返回)
#         self.sdf_out     = nn.Linear(hidden_dim, 1)
#         self.feature_out = nn.Linear(hidden_dim, hidden_dim)  # 保留以备后续使用

#     # position encoding
#     def pos_enc(self, x: torch.Tensor) -> torch.Tensor:
#         enc = [x]
#         for i in range(self.pe_freqs):
#             if self.pe_mask[i]:
#                 freq = (2.0 ** i) * math.pi
#                 enc.append(torch.sin(freq * x))
#                 enc.append(torch.cos(freq * x))
#             else:
#                 enc.append(torch.zeros_like(x))
#                 enc.append(torch.zeros_like(x))
#         return torch.cat(enc, dim=-1)

#     # def pos_enc(self, x: torch.Tensor) -> torch.Tensor:
#     #     enc = [x]
#     #     for i in range(self.pe_freqs):
#     #         freq = (2.0 ** i) * math.pi
#     #         enc.append(torch.sin(freq * x))
#     #         enc.append(torch.cos(freq * x))
#     #     return torch.cat(enc, dim=-1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#           x: [N,6] = [xyz, rgb]
#         Returns:
#           d: [N] predicted signed distance
#         """
#         # 拆分位置和颜色（后者暂不参与 SDF 计算）
#         p = x[:, :3]  # [N,3]
#         rgb = x[:,3:]

#         # 对 xyz 做 PE 并通过 MLP
#         pe = self.pos_enc(p)  # [N, input_dim]
#         input_with_color = torch.cat([pe, rgb], dim=-1)  # 加上顏色資訊

#         h = input_with_color
#         for i in range(self.num_layers):
#             if i == self.skip:
#                 h = torch.cat([h, input_with_color], dim=-1)
#             lin = self.layers[2*i]
#             act = self.layers[2*i + 1]
#             h = act(lin(h))

#         d = self.sdf_out(h).squeeze(-1)  # [N]
#         # geometry feature g = self.feature_out(h)  # [N, hidden_dim], 暂不返回

#         return d
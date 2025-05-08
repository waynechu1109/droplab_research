import torch
import torch.nn as nn
import tinycudann as tcnn  # Instant-NGP 的 CUDA 加速 hash encoding 實現
import math

# model architecture
class SDFNet(nn.Module):
    def __init__(self, in_dim=6): # input dimension got 6 dimensions (x, y, z, r, g, b)
        super(SDFNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Output shape: [N]



    # def __init__(self, input_dim=6, hidden_dim=128, num_layers=8, skip_connection_at=4, pe_freqs=6):
    #     super().__init__()
    #     self.num_layers = num_layers
    #     self.skip_connection_at = skip_connection_at
    #     self.pe_freqs = pe_freqs

    #     self.input_dim = 3 + 2 * pe_freqs * 3 + 3  # PE(xyz) + RGB = final input dim
    #     layers = []
    #     for i in range(num_layers):
    #         if i == 0:
    #             in_dim = self.input_dim
    #         elif i == skip_connection_at:
    #             in_dim = hidden_dim + self.input_dim  # skip connection
    #         else:
    #             in_dim = hidden_dim

    #         layers.append(nn.Linear(in_dim, hidden_dim))
    #         layers.append(nn.Softplus(beta=100))
    #     self.hidden_layers = nn.ModuleList(layers)
    #     self.output_layer = nn.Linear(hidden_dim, 1)

    # def pos_enc(self, x):
    #     enc = [x]
    #     for i in range(self.pe_freqs):
    #         enc.append(torch.sin((2 ** i) * math.pi * x))
    #         enc.append(torch.cos((2 ** i) * math.pi * x))
    #     return torch.cat(enc, dim=-1)

    # def forward(self, x):  # x: [N, 6] = [xyz, rgb]
    #     pos = x[:, :3]                  # [N, 3]
    #     color = x[:, 3:]                # [N, 3]
    #     pos_encoded = self.pos_enc(pos)  # [N, 3 + 2*pe_freqs*3]
    #     net_input = torch.cat([pos_encoded, color], dim=-1)  # [N, input_dim]

    #     h = net_input
    #     for i in range(self.num_layers):
    #         lin = self.hidden_layers[i * 2]
    #         act = self.hidden_layers[i * 2 + 1]
    #         if i == self.skip_connection_at:
    #             h = torch.cat([h, net_input], dim=-1)
    #         h = act(lin(h))

    #     return self.output_layer(h).squeeze(-1)


    # def __init__(self):
    #     super().__init__()
    #     self.encoder = tcnn.NetworkWithInputEncoding(
    #         n_input_dims=3,  # x, y, z  # concatenate color later
    #         n_output_dims=64,
    #         encoding_config={
    #             "otype": "HashGrid",
    #             "n_levels": 16,
    #             "n_features_per_level": 2,
    #             "log2_hashmap_size": 19,
    #             "base_resolution": 16,
    #             "per_level_scale": 2.0
    #         },
    #         network_config={
    #             "otype": "FullyFusedMLP",
    #             "activation": "ReLU",
    #             "output_activation": "None",
    #             "n_neurons": 64,
    #             "n_hidden_layers": 2
    #         }
    #     )
    #     # 顏色會直接與 hash encoding 特徵拼接
    #     self.color_fc = nn.Sequential(
    #         nn.Linear(64 + 3, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 1)  # SDF 預測
    #     )

    # def forward(self, x):
    #     pos = x[:, :3]  # x, y, z
    #     color = x[:, 3:]  # r, g, b
    #     encoded = self.encoder(pos)
    #     features = torch.cat([encoded, color], dim=-1)
    #     return self.color_fc(features).squeeze(-1)

    # def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, per_level_scale=2.0):
    #     super().__init__()
    #     self.n_levels = n_levels
    #     self.n_features_per_level = n_features_per_level
    #     self.hashmap_size = 2 ** log2_hashmap_size
    #     self.base_resolution = base_resolution
    #     self.per_level_scale = per_level_scale

    #     # 建立 hash grid embedding tables
    #     self.embeddings = nn.ModuleList([
    #         nn.Embedding(self.hashmap_size, n_features_per_level)
    #         for _ in range(n_levels)
    #     ])
    #     for emb in self.embeddings:
    #         nn.init.uniform_(emb.weight, a=-1e-4, b=1e-4)

    #     # SDF MLP (接上 hash feature + color)
    #     total_feat_dim = n_levels * n_features_per_level + 3  # 3 是 RGB
    #     self.color_fc = nn.Sequential(
    #         nn.Linear(total_feat_dim, 64),
    #         nn.Tanh(),
    #         nn.Linear(64, 32),
    #         nn.Tanh(),
    #         nn.Linear(32, 1)  # 輸出 SDF
    #     )

    # def hash(self, x_int):
    #     # x_int: [N, 3]，取整後的 voxel index
    #     primes = torch.tensor([1, 2654435761, 805459861], device=x_int.device)
    #     hashed = (x_int * primes).sum(dim=-1) & (self.hashmap_size - 1)
    #     return hashed  # [N]

    # def encode(self, x):  # x: [N, 3], in [-1, 1]
    #     x = (x + 1) / 2  # 轉換成 [0, 1]
    #     feats = []
    #     for i in range(self.n_levels):
    #         res = int(self.base_resolution * (self.per_level_scale ** i))
    #         x_rescaled = (x * res).floor().clamp(0, res - 1).long()  # [N,3]
    #         h = self.hash(x_rescaled)  # [N]
    #         feats.append(self.embeddings[i](h))  # [N, F]
    #     return torch.cat(feats, dim=-1)  # [N, F_total]

    # def forward(self, x):  # x: [N, 6] -> [x,y,z, r,g,b]
    #     pos = x[:, :3]
    #     color = x[:, 3:]
    #     encoded = self.encode(pos)
    #     features = torch.cat([encoded, color], dim=-1)
    #     sdf = self.color_fc(features).squeeze(-1)
    #     return sdf

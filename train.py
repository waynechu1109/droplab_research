import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy as np
from tqdm import tqdm

epochs = 5000
desc = "weight:(20,1,0.5)"

pointcloud_path = "data/output_pointcloud_all.ply"
log_path = f"log/{epochs}_{desc}.txt"

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# read pointcloud
pcd = o3d.io.read_point_cloud(pointcloud_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
x = torch.tensor(np.hstack((points, colors)), dtype=torch.float32)  # [N,6], x includes x and c
x = x.to(device)

# generate point with noises
sigma = 0.01
epsilon = torch.randn_like(x[:, :3]) * sigma # noise
x_noisy = x[:, :3] + epsilon  # [N,3]
x_noisy_full = torch.cat([x_noisy, x[:, 3:]], dim=1)  # add the color of x to x^n -> [N,6]

# Move noisy inputs and epsilon to device
x_noisy_full = x_noisy_full.to(device)
epsilon = epsilon.to(device)

# model architecture (多層感知機)
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

model = SDFNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# total loss calculation
def compute_loss(model, x, x_noisy_full, epsilon):
    f_x = model(x)              # f(x, c) 應接收完整 6 維輸入
    f_x_noisy = model(x_noisy_full)

    # Part 1: MSE between predicted SDF at xⁿ and actual |ε|
    true_dist = epsilon.norm(dim=1)
    loss_sdf = F.mse_loss(f_x_noisy, true_dist)

    # Part 2: ||f(x, c)||
    loss_zero = f_x.abs().mean()

    # Part 3: Eikonal: ||∇f(x^n, c) - 1||
    x_noisy_full.requires_grad_() # enable partial differentiation
    f_pred = model(x_noisy_full)
    grads = torch.autograd.grad(
        outputs=f_pred,       # f(xⁿ, c)
        inputs=x_noisy_full,  # input with 6 dimensions: [x, y, z, r, g, b]
        grad_outputs=torch.ones_like(f_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, :3]  # take the gradieant w.r.t. (x, y, z)

    grad_norm = grads.norm(dim=1)
    loss_eikonal = ((grad_norm - 1) ** 2).mean()

    return loss_sdf, loss_zero, loss_eikonal

# training example
model.train()
pbar = tqdm(range(epochs), desc="Training", ncols=100)

with open(log_path, "w") as f:
    f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal\n")

for epoch in pbar:
    optimizer.zero_grad()
    loss_sdf, loss_zero, loss_eikonal = compute_loss(model, x, x_noisy_full, epsilon)
    loss_total = 20 * loss_sdf + 1 * loss_zero + 0.5 * loss_eikonal
    loss_total.backward()
    optimizer.step()

    pbar.set_postfix(loss=loss_total.item())

    # log each component
    with open(log_path, "a") as f:
        f.write(f"{epoch},{loss_total.item():.6f},{loss_sdf.item():.6f},{loss_zero.item():.6f},{loss_eikonal.item():.6f}\n")

torch.save(model.state_dict(), f"ckpt/sdf_model_{epochs}_{desc}.pt")
print("Training finished.")

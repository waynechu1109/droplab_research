import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
from model import SDFNet
from loss import compute_loss
from torch.optim.lr_scheduler import LambdaLR

warmup = False
epochs = 5000
warmup_epochs = 500
lr = 0.005
sigma = 0.01

desc = "pointnetPE_n0.01_weight:(20,1,0.5)_v2"

pointcloud_path = "data/output_pointcloud_1.ply"
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

# generate point with noises (Sampling)
epsilon = torch.randn_like(x[:, :3]) * sigma # noise
x_noisy = x[:, :3] + epsilon  # [N,3]
x_noisy_full = torch.cat([x_noisy, x[:, 3:]], dim=1)  # add the color of x to x^n -> [N,6]

# Move noisy inputs and epsilon to device
x_noisy_full = x_noisy_full.to(device)
epsilon = epsilon.to(device)

# model = SDFNet().to(device)
model = SDFNet(pe_freqs=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# warmup
if warmup:
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs)
    )

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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip norm
    optimizer.step()
    if warmup:
        scheduler.step() # warming up

    pbar.set_postfix(loss=loss_total.item())

    # log each component
    with open(log_path, "a") as f:
        f.write(f"{epoch},{loss_total.item():.6f},{loss_sdf.item():.6f},{loss_zero.item():.6f},{loss_eikonal.item():.6f}\n")

torch.save(model.state_dict(), f"ckpt/sdf_model_{epochs}_{desc}.pt")
print("Training finished.")

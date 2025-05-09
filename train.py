import argparse
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
from model import SDFNet
from loss import compute_loss
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description="SDFNet training script.")
parser.add_argument('--epochs', type=int, default=5000, help="Number of training epochs.")
parser.add_argument('--lr', type=float, default=0.005, help="Learning rate.")
parser.add_argument('--sigma', type=float, default=0.01, help="Noise sigma.")
parser.add_argument('--desc', type=str, required=True, help="Experiment description.")
parser.add_argument('--log_path', type=str, required=True, help="Log file path.")
parser.add_argument('--ckpt_path', type=str, required=True, help="Checkpoint save path.")
args = parser.parse_args()

lr_tune = True
epochs = args.epochs
lr_tune_epochs = 500
lr = args.lr
sigma = args.sigma

desc = args.desc
log_path = args.log_path
ckpt_path = args.ckpt_path

pointcloud_path = "data/output_pointcloud_shoes.ply"

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

# lr_tune
if lr_tune:
    scheduler = MultiStepLR(
    optimizer,
    milestones=[epochs*0.2, epochs*0.4, epochs*0.6, epochs*0.8],
    gamma=0.5
)
# training example
model.train()
pbar = tqdm(range(epochs), desc="Training", ncols=100)

with open(log_path, "w") as f:
    f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal\n")

for epoch in pbar:
    optimizer.zero_grad()
    loss_sdf, loss_zero, loss_eikonal = compute_loss(model, x, x_noisy_full, epsilon)
    loss_total = 10 * loss_sdf + 1 * loss_zero + 0.05 * loss_eikonal
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip norm
    optimizer.step()
    if lr_tune:
        scheduler.step() # warming up

    pbar.set_postfix(
        loss=loss_total.item(),
        lr=optimizer.param_groups[0]['lr']
    )

    # log each component
    with open(log_path, "a") as f:
        f.write(f"{epoch},{loss_total.item():.6f},{loss_sdf.item():.6f},{loss_zero.item():.6f},{loss_eikonal.item():.6f}\n")

torch.save(model.state_dict(), ckpt_path)
print("Training finished.")

import json

def load_schedule(schedule_path):
    with open(schedule_path, 'r') as f:
        return json.load(f)

def get_stage_config(schedule, epoch):
    if epoch < schedule['coarse']['epochs']:
        return schedule['coarse']
    else:
        return schedule['fine']

import argparse
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
from model import SDFNet
from loss import compute_loss
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR

parser = argparse.ArgumentParser(description="SDFNet training script.")
# parser.add_argument('--epochs', type=int, default=5000, help="Number of training epochs.")
parser.add_argument('--lr', type=float, default=0.005, help="Learning rate.")
# parser.add_argument('--sigma', type=float, default=0.01, help="Noise sigma.")
parser.add_argument('--desc', type=str, required=True, help="Experiment description.")
# parser.add_argument('--para', type=float, required=True, help="Parameter want to control.")
parser.add_argument('--log_path', type=str, required=True, help="Log file path.")
parser.add_argument('--ckpt_path', type=str, required=True, help="Checkpoint save path.")
parser.add_argument('--file_name', type=str, required=True, help="Pointcloud file name.")
parser.add_argument('--schedule_path', type=str, required=True, help="Training schedule file name.")
args = parser.parse_args()

# epochs = args.epochs
lr = args.lr
# sigma = args.sigma
# para = args.para
file_name = args.file_name

desc = args.desc
log_path = args.log_path
ckpt_path = args.ckpt_path
sche_path = args.schedule_path

# Load training schedule
schedule = load_schedule(sche_path)

total_epochs = schedule["total_epochs"]

# # PE setting
# pe_max = int(para)          # max freq index
# pe_min = 0                  # start from 0
# pe_ramp = 800              # total epochs to fully activate all

pointcloud_path = f"data/output_pointcloud_{file_name}_normal.ply"

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# read pointcloud
pcd = o3d.io.read_point_cloud(pointcloud_path)


# get everything needed
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
normals = np.asarray(pcd.normals)

x = torch.tensor(np.hstack((points, colors)), dtype=torch.float32)  # [N,6], x includes x and c
x = x.to(device)

assert np.allclose(np.asarray(pcd.points), x[:, :3].cpu().numpy(), atol=1e-5), "!!!!Mismatch between x and normals!!!!"

# # generate point with noises (Sampling)
# epsilon = torch.randn_like(x[:, :3]) * sigma # noise
# x_noisy = x[:, :3] + epsilon  # [N,3]
# x_noisy_full = torch.cat([x_noisy, x[:, 3:]], dim=1)  # add the color of x to x^n -> [N,6]

# # Move noisy inputs and epsilon to device
# x_noisy_full = x_noisy_full.to(device)
# epsilon = epsilon.to(device)

model = SDFNet(pe_freqs=schedule["fine"]["pe_freqs"]).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=1e-2
)

# lr_tune
# if lr_tune:
# scheduler = MultiStepLR(
#     optimizer,
#     milestones=[epochs*0.2, epochs*0.4, epochs*0.6, epochs*0.8],
#     gamma=0.5
# )
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_epochs,      
    eta_min=1e-5             # lowest lr
)
# scheduler = OneCycleLR(
#     optimizer,
#     max_lr=lr,
#     epochs=total_epochs,
#     steps_per_epoch=1,
#     pct_start=0.1,
#     anneal_strategy="cos",
#     final_div_factor=10
# )

# training
model.train()
pbar = tqdm(range(total_epochs), desc="Training", ncols=100)

with open(log_path, "w") as f:
    # f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal,loss_edge,loss_normal\n")
    # f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal,loss_normal\n")
    # f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal,loss_normal\n")
    # f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal,loss_normal,loss_consistency\n")
    f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal,loss_normal,loss_consistency,learning_rate\n")

for epoch in pbar:
    # Set the training configuration
    stage_cfg = get_stage_config(schedule, epoch)

    sigma = stage_cfg["sigma"]
    pe_max = stage_cfg["pe_freqs"]
    pe_ramp = stage_cfg["pe_ramp"]

    # get the weight information from schedule
    weight_sdf           = stage_cfg["loss_weights"]["loss_sdf"]
    weight_zero          = stage_cfg["loss_weights"]["loss_zero"]

    eik_init             = stage_cfg["loss_weights"]["loss_eikonal"]["loss_eikonal_init"]
    weight_eikonal_final = stage_cfg["loss_weights"]["loss_eikonal"]["loss_eikonal_final"]
    eik_ramp             = stage_cfg["loss_weights"]["loss_eikonal"]["loss_eikonal_ramp"]
    
    weight_normal        = stage_cfg["loss_weights"]["loss_normal"]
    weight_consistency   = stage_cfg["loss_weights"]["loss_consistency"]

    if stage_cfg is schedule["coarse"]:
        epoch_in_stage = epoch
        pe_min = 0
    else:
        epoch_in_stage = epoch - schedule["coarse"]["epochs"]
        pe_min = schedule["coarse"]["pe_freqs"]

    # generate point with noises (Sampling)
    epsilon = torch.randn_like(x[:, :3]) * sigma # noise
    x_noisy = x[:, :3] + epsilon  # [N,3]
    x_noisy_full = torch.cat([x_noisy, x[:, 3:]], dim=1)  # add the color of x to x^n -> [N,6]

    # Move noisy inputs and epsilon to device
    x_noisy_full = x_noisy_full.to(device)
    epsilon = epsilon.to(device)

    # === Progressive PE Mask Update ===
    # alpha = 1 / (1 + np.exp(-(epoch_in_stage - pe_ramp/2) / (pe_ramp/10)))
    # active_pe = pe_min + int((pe_max - pe_min) * alpha)
    active_pe = pe_min + int((pe_max - pe_min) * min(epoch_in_stage / pe_ramp, 1.0))
    pe_mask = torch.zeros(model.pe_freqs, dtype=torch.bool)
    pe_mask[:active_pe] = True
    model.pe_mask = pe_mask.to(device)

    optimizer.zero_grad()
    # loss_sdf, loss_zero, loss_eikonal, loss_edge, loss_normal = compute_loss(model, x, x_noisy_full, epsilon, normals)
    # loss_sdf, loss_zero, loss_eikonal, loss_normal = compute_loss(model, x, x_noisy_full, epsilon, normals)
    loss_sdf, loss_zero, loss_eikonal, loss_normal, loss_consistency = compute_loss(model, x, x_noisy_full, epsilon, normals)

    # -------------------weight setting--------------------
    # loss_total = 5 * loss_sdf + 0.5 * loss_zero + 0.05 * loss_eikonal
    # eik_init, eik_final, ramp = 0.01, 0.05, 750
    # w_eik = eik_init + (eik_final - eik_init) * min(epoch/ramp, 1.0)
    # loss_total = 4.2 * loss_sdf \
    #         + 0.5 * loss_zero \
    #         + w_eik * loss_eikonal \
    #         + 0.1 * loss_edge \
    #         + 0.05 * loss_normal

    # loss_total = 4.2 * loss_sdf \
    #         + 0.5 * loss_zero \
    #         + w_eik * loss_eikonal \
    #         + 0.05 * loss_normal 

    w_eik = eik_init + (weight_eikonal_final - eik_init) * min(epoch / eik_ramp, 1.0)
    loss_total = weight_sdf * loss_sdf \
                + weight_zero * loss_zero \
                + w_eik * loss_eikonal \
                + weight_normal * loss_normal \
                + weight_consistency * loss_consistency
    
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip norm
    optimizer.step() # update model's parameters
    scheduler.step() # update lr

    pbar.set_postfix(
        loss=loss_total.item(),
        lr=optimizer.param_groups[0]['lr']
    )

    current_lr = optimizer.param_groups[0]['lr']
    # log each component
    with open(log_path, "a") as f:
        # f.write(f"{epoch},{loss_total.item():.6f},{loss_sdf.item():.6f},{loss_zero.item():.6f},{loss_eikonal.item():.6f},{loss_edge.item():.6f},{loss_normal.item():.6f}\n")
        # f.write(f"{epoch},{loss_total.item():.6f},{loss_sdf.item():.6f},{loss_zero.item():.6f},{loss_eikonal.item():.6f},{loss_normal.item():.6f}\n")
        # f.write(f"{epoch},{loss_total.item():.6f},{loss_sdf.item():.6f},{loss_zero.item():.6f},{loss_eikonal.item():.6f},{loss_normal.item():.6f},{loss_consistency.item():.6f}\n")
        f.write(f"{epoch},{loss_total.item():.6f},{loss_sdf.item():.6f},{loss_zero.item():.6f},{loss_eikonal.item():.6f},{loss_normal.item():.6f},{loss_consistency.item():.6f},{current_lr:.8f}\n")

# torch.save(model.state_dict(), ckpt_path)
torch.save({
    "model_state_dict": model.state_dict(),
    "pe_freqs": pe_max
}, ckpt_path)
print("Training finished.")

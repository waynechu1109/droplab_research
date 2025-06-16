import json
from PIL import Image
import glob
import argparse
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
from model import SDFNet
from loss import compute_loss
import matplotlib.pyplot as plt
import torchvision.transforms as T
from copy import deepcopy
from utils import str2bool

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

def load_schedule(schedule_path):
    with open(schedule_path, 'r') as f:
        return json.load(f)

def get_stage_config(schedule, epoch):
    if epoch < schedule['coarse']['epochs']:
        return schedule['coarse']
    elif epoch < schedule["coarse"]["epochs"] + schedule["fine"]["epochs"]:
        return schedule['fine']

parser = argparse.ArgumentParser(description="SDFNet training script.")
parser.add_argument('--lr', type=float, default=0.005, help="Learning rate.")
parser.add_argument('--desc', type=str, required=True, help="Experiment description.")
parser.add_argument('--log_path', type=str, required=True, help="Log file path.")
parser.add_argument('--ckpt_path', type=str, required=True, help="Checkpoint save path.")
parser.add_argument('--file_name', type=str, required=True, help="Pointcloud file name.")
parser.add_argument('--schedule_path', type=str, required=True, help="Training schedule file name.")
parser.add_argument('--is_a100', type=str2bool, required=True, help="Training on A100 or not.")
# parser.add_argument('--para', type=float, required=True, help="Parameter want to control.")
args = parser.parse_args()

lr = args.lr
is_a100 = args.is_a100
# para = args.para
file_name = args.file_name

desc = args.desc
log_path = args.log_path
ckpt_path = args.ckpt_path
sche_path = args.schedule_path

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load training schedule
schedule = load_schedule(sche_path)
total_epochs = schedule["total_epochs"]

# Read pcd
pointcloud_path = f"data/output_pointcloud_{file_name}_normal.ply"
pcd = o3d.io.read_point_cloud(pointcloud_path)
# get everything needed
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
normals = np.asarray(pcd.normals)

points = torch.tensor(points, dtype=torch.float32, device=device)
colors = torch.tensor(colors, dtype=torch.float32, device=device)

# Read info json
with open(f"data/pointcloud_{file_name}_normal_info.json") as f:
    info = json.load(f)
poses = torch.tensor(info["poses"], dtype=torch.float32, device=device)
focals = torch.tensor(info["focals"], dtype=torch.float32, device=device)
centre = torch.tensor(info["centre"], dtype=torch.float32, device=device)
scale = torch.tensor(info["scale"], dtype=torch.float32, device=device)

# build intrinsic K
image_candidates = glob.glob(f"data/{file_name}.jpg") + glob.glob(f"data/{file_name}.png")
if len(image_candidates) == 0:
    raise FileNotFoundError(f"No image found for {file_name}.jpg or .png in data/")
print("Successfully read input image.")
img_path = image_candidates[0] 
with Image.open(img_path) as img:
    W, H = img.size
    img_tensor = T.ToTensor()(img).permute(1, 2, 0).to(device)
cx, cy = W / 2, H / 2
fx = fy = focals[0][0]
K = torch.tensor([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0, 1]
], device=device)

# === Step 1: 原始 pose ===
pose_0 = poses[0].clone()
R = pose_0[:3, :3]  # [3,3]
t = pose_0[:3, 3]   # [3,]

# === Step 2: normalize 相機位置 ===
cam_pos = (-R.T @ t)                # camera position in world space
cam_pos_norm = (cam_pos - centre) / scale  # normalize

# === Step 3: 計算新的 t ===
t_norm = -R @ cam_pos_norm          # 推回相機座標系下的 t'

# === Step 4: 建立新 pose ===
pose_0_normed = torch.eye(4, device=device)
pose_0_normed[:3, :3] = R
pose_0_normed[:3, 3] = t_norm

# === 使用新 pose ===
cam_pose_0 = pose_0_normed

# render
# pcd is already normalized
print("Rendering in normalized space...")
print("Cam pose_0:\n", cam_pose_0)
if isinstance(points, torch.Tensor):
    print("Points range:", torch.amin(points, dim=0), torch.amax(points, dim=0))
else:
    print("Points range:", np.min(points, axis=0), np.max(points, axis=0))

# gt_image, dep = render_pointcloud(points, colors, cam_pose, K, (H, W))
gt_image = img_tensor
print("gt_image stats:", gt_image.min().item(), gt_image.max().item())
print("Any pixels rendered:", (gt_image > 0).sum().item())

gt_image_np = gt_image.detach().cpu().numpy()
print(f'save debug img.')
plt.imsave("debug_gt_image.png", gt_image_np)


x = torch.cat([points, colors], dim=1) # [N,6], x includes x and c
# x = points
x = x.to(device)

assert np.allclose(np.asarray(pcd.points), x[:, :3].cpu().numpy(), atol=1e-5), "!!!!Mismatch between x and normals!!!!"

max_pe_freqs = max(schedule["coarse"]["pe_freqs"],
                   schedule["fine"]["pe_freqs"])

print(f'max_pe_freqs: {max_pe_freqs}')

model = SDFNet(pe_freqs=max_pe_freqs).to(device)
model.scale = scale
model.centre = centre
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=1e-2
)

# lr_tune
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_epochs,      
    eta_min= lr / 500.0            # lowest lr
)

scaler = GradScaler()

# 初始 scheduler（非 color 階段）
# scheduler_1 = CosineAnnealingLR(
#     optimizer,
#     T_max=schedule["coarse"]["epochs"] + schedule["fine"]["epochs"],
#     eta_min=1e-5
# )
# scheduler_2 = None  # 預設為 None，color 階段才會建立

# training
model.train()
pbar = tqdm(range(total_epochs), desc="Training")

with open(log_path, "w") as f:
    f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal,loss_normal,loss_sparse,loss_color_geo,learning_rate\n")

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
    # weight_consistency   = stage_cfg["loss_weights"]["loss_consistency"]

    weight_sparse        = stage_cfg["loss_weights"]["loss_sparse"]
    # weight_render        = stage_cfg["loss_weights"]["loss_render"]
    # weight_color_smooth  = stage_cfg["loss_weights"]["loss_color_smooth"]
    weight_color_geo  = stage_cfg["loss_weights"]["loss_color_geo"]

    if stage_cfg is schedule["coarse"]:
        epoch_in_stage = epoch
        pe_min = 0
    elif stage_cfg is schedule["fine"]:
        epoch_in_stage = epoch - schedule["coarse"]["epochs"]
        pe_min = schedule["coarse"]["pe_freqs"]


    # if stage_cfg is schedule["color"]:
    #     for name, param in model.named_parameters():
    #         param.requires_grad = "rgb_head" in name

    # # print(f'max. lr: {lr}')
    # if stage_cfg is schedule["color"] and scheduler_2 is None:
    #     print(f"Switching to second CosineAnnealingLR scheduler for color stage, lr={lr}")
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 1e-2*lr
    #     scheduler_2 = CosineAnnealingLR(
    #         optimizer,
    #         T_max=schedule["color"]["epochs"],
    #         eta_min=1e-2*1e-5
    #     )

    # generate 6D noise and apply to all (x, y, z, r, g, b)
    epsilon = torch.randn_like(x) * sigma   # [N,6]
    x_noisy_full = x + epsilon              # [N,6]

    # Move noisy inputs and epsilon to device
    x_noisy_full = x_noisy_full.to(device)
    epsilon = epsilon.to(device)

    # === Progressive PE Mask Update ===
    active_pe = pe_min + int((pe_max - pe_min) * min(epoch_in_stage / pe_ramp, 1.0))
    pe_mask = torch.zeros(model.pe_freqs, dtype=torch.bool)
    pe_mask[:active_pe] = True

    # print(f'active_pe: {active_pe}')
    model.pe_mask = pe_mask.to(device)

    optimizer.zero_grad()

    with autocast():
        loss_sdf, loss_zero, loss_eikonal, loss_normal, loss_sparse, loss_color_geo = compute_loss(
            model,
            x,
            x_noisy_full,
            epsilon,
            normals,
            H,
            W,
            K,
            cam_pose_0,
            gt_image,
            is_a100,
            weight_sdf,
            weight_zero,
            eik_init,
            weight_normal,
            weight_sparse,
            weight_color_geo
        )

        # -------------------weight setting--------------------
        w_eik = eik_init + (weight_eikonal_final - eik_init) * min(epoch / eik_ramp, 1.0)
        loss_total = weight_sdf * loss_sdf \
                    + weight_zero * loss_zero \
                    + w_eik * loss_eikonal \
                    + weight_normal * loss_normal \
                    + weight_sparse * loss_sparse \
                    + weight_color_geo * loss_color_geo
                    
    scaler.scale(loss_total).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip norm
    scaler.step(optimizer) # update model's parameters
    scaler.update()
    scheduler.step() # update lr
    # if scheduler_2 is not None:
    #     scheduler_2.step()
    # else:
    #     scheduler_1.step()

    pbar.set_postfix(
        loss=loss_total.item(),
        lr=optimizer.param_groups[0]['lr']
    )

    current_lr = optimizer.param_groups[0]['lr']
    # log each component
    with open(log_path, "a") as f:
        f.write(f"{epoch}, \
                {loss_total.item():.6f}, \
                {loss_sdf.item():.6f}, \
                {loss_zero.item():.6f}, \
                {loss_eikonal.item():.6f}, \
                {loss_normal.item():.6f}, \
                {loss_sparse.item():.6f}, \
                {loss_color_geo.item():.6f}, \
                {current_lr:.8f}\n")

    if epoch % 500 == 0 and epoch >= 1000:
        print(f'saving pe_freqs @ epoch {epoch}: {model.pe_freqs}')
        torch.save({
                "model_state_dict": model.state_dict(),
                "pe_freqs": model.pe_freqs
        }, f"{ckpt_path}_epoch{epoch}.pt")

torch.save({
            "model_state_dict": model.state_dict(),
            "pe_freqs": model.pe_freqs
        }, f"{ckpt_path}_final.pt")
print("Training finished.")
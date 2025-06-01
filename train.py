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

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR

def load_schedule(schedule_path):
    with open(schedule_path, 'r') as f:
        return json.load(f)

def get_stage_config(schedule, epoch):
    if epoch < schedule['coarse']['epochs']:
        return schedule['coarse']
    else:
        return schedule['fine']

def render_pointcloud(points: torch.Tensor,
                      colors: torch.Tensor,
                      cam_pose: torch.Tensor,
                      K: torch.Tensor,
                      image_size: tuple[int, int]
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    device = points.device
    H, W = image_size

    n = points.shape[0]
    pts_h = torch.cat([points, torch.ones(n, 1, device=device)], dim=1)  # (n,4)
    cam_h = (cam_pose @ pts_h.T).T
    Xc, Yc, Zc = cam_h[:, 0], cam_h[:, 1], cam_h[:, 2]

    proj = (K @ cam_h[:, :3].T).T
    u = (proj[:, 0] / proj[:, 2]).round().long()
    v = (proj[:, 1] / proj[:, 2]).round().long()

    valid = (Zc > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Zc, cols = u[valid], v[valid], Zc[valid], colors[valid]

    image = torch.zeros((H, W, 3), device=device)
    depth = torch.full((H, W), float('inf'), device=device)

    for xi, yi, zi, ci in zip(u, v, Zc, cols):
        if zi < depth[yi, xi]:
            depth[yi, xi] = zi
            image[yi, xi] = ci

    return image, depth

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="SDFNet training script.")
parser.add_argument('--lr', type=float, default=0.005, help="Learning rate.")
parser.add_argument('--desc', type=str, required=True, help="Experiment description.")
parser.add_argument('--log_path', type=str, required=True, help="Log file path.")
parser.add_argument('--ckpt_path', type=str, required=True, help="Checkpoint save path.")
parser.add_argument('--file_name', type=str, required=True, help="Pointcloud file name.")
parser.add_argument('--schedule_path', type=str, required=True, help="Training schedule file name.")
parser.add_argument('--is_a100', type=str2bool, required=True, help="Training on A100 or not.")
parser.add_argument('--para', type=float, required=True, help="Parameter want to control.")
args = parser.parse_args()

lr = args.lr
is_a100 = args.is_a100
para = args.para
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
cx, cy = W / 2, H / 2
fx = fy = focals[0][0]
K = torch.tensor([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0, 1]
], device=device)

# take the 1st pose
pose = poses[0]

# convert points/colors to torch
points = torch.tensor(points, dtype=torch.float32, device=device)
colors = torch.tensor(colors, dtype=torch.float32, device=device)
cam_pose = pose.clone().detach().to(device)

# render
unnorm_points = points * scale + centre
gt_image, dep = render_pointcloud(unnorm_points, colors, cam_pose, K, (H, W))
gt_image_np = gt_image.detach().cpu().numpy()
print(f'save debug img.')
plt.imsave("debug_gt_image.png", gt_image_np)


# x = torch.cat([points, colors], dim=1) # [N,6], x includes x and c
x = points
x = x.to(device)

assert np.allclose(np.asarray(pcd.points), x[:, :3].cpu().numpy(), atol=1e-5), "!!!!Mismatch between x and normals!!!!"

model = SDFNet(pe_freqs=schedule["fine"]["pe_freqs"]).to(device)
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
    eta_min=1e-5             # lowest lr
)

# training
model.train()
pbar = tqdm(range(total_epochs), desc="Training")

with open(log_path, "w") as f:
    f.write("epoch,loss_total,loss_sdf,loss_zero,loss_eikonal,loss_normal,loss_sparse,loss_render,learning_rate\n")

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
    weight_render        = para

    if stage_cfg is schedule["coarse"]:
        epoch_in_stage = epoch
        pe_min = 0
        weight_render = 0.0
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
    active_pe = pe_min + int((pe_max - pe_min) * min(epoch_in_stage / pe_ramp, 1.0))
    pe_mask = torch.zeros(model.pe_freqs, dtype=torch.bool)
    pe_mask[:active_pe] = True
    model.pe_mask = pe_mask.to(device)

    optimizer.zero_grad()
    loss_sdf, loss_zero, loss_eikonal, loss_normal, loss_sparse, loss_render = compute_loss(model, 
                                                                                                 x, 
                                                                                                 x_noisy_full, 
                                                                                                 epsilon, 
                                                                                                 normals,
                                                                                                 H,
                                                                                                 W,
                                                                                                 K,
                                                                                                 cam_pose,
                                                                                                 gt_image,
                                                                                                 is_a100)

    # -------------------weight setting--------------------
    w_eik = eik_init + (weight_eikonal_final - eik_init) * min(epoch / eik_ramp, 1.0)
    loss_total = weight_sdf * loss_sdf \
                + weight_zero * loss_zero \
                + w_eik * loss_eikonal \
                + weight_normal * loss_normal \
                + weight_sparse * loss_sparse \
                + weight_render * loss_render
    
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
        f.write(f"{epoch}, \
                {loss_total.item():.6f}, \
                {loss_sdf.item():.6f}, \
                {loss_zero.item():.6f}, \
                {loss_eikonal.item():.6f}, \
                {loss_normal.item():.6f}, \
                {loss_sparse.item():.6f}, \
                {loss_render.item():.6f}, \
                {current_lr:.8f}\n")

# torch.save(model.state_dict(), ckpt_path)
torch.save({
    "model_state_dict": model.state_dict(),
    "pe_freqs": pe_max
}, ckpt_path)
print("Training finished.")
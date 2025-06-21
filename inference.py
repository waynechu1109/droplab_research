import argparse
import torch
import numpy as np
from skimage import measure
import open3d as o3d
import json
from model import SDFNet
from tqdm import tqdm
import matplotlib.pyplot as plt 
from PIL import Image
import glob
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

from utils import render_pointcloud, str2bool

hist_ = False
slice = True
render_verts = False

parser = argparse.ArgumentParser(description="SDFNet inference script (KNN Color, Fast).")
parser.add_argument('--res', type=int, default=256, help="Voxel grid resolution.")
parser.add_argument('--ckpt_path', type=str, required=True, help="Path to model checkpoint.")
parser.add_argument('--output_mesh', type=str, required=True, help="Output mesh file path.")
parser.add_argument('--file_name', type=str, required=True, help="Pointcloud file name.")
args = parser.parse_args()

res = args.res
ckpt_path = args.ckpt_path
output_mesh = args.output_mesh
file_name = args.file_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Read point cloud and normalization info ===
pcd = o3d.io.read_point_cloud(f"data/output_pointcloud_{file_name}_normal.ply")
with open(f"data/pointcloud_{file_name}_normal_info.json", "r") as f:
    info = json.load(f)
centre = np.array(info["centre"])
scale = info["scale"]

# Get point cloud xyz/rgb
pc_xyz = np.asarray(pcd.points)   # (N, 3)
pc_rgb = np.asarray(pcd.colors)   # (N, 3)

# === Load model ===
checkpoint = torch.load(ckpt_path, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    pe_freqs = checkpoint.get("pe_freqs")
    model = SDFNet(pe_freqs=pe_freqs).to(device)
    model.pe_mask = torch.ones(pe_freqs, dtype=torch.bool).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded successfully with pe_freqs = {pe_freqs}")
else:
    raise ValueError("Fail to load model: incorrect format")

model.eval()

# === Generate voxel grid in normalized space ===
min_bound = [-1.0, -1.0, -1.0]
max_bound = [1.0, 1.0, 1.0]
x_vals = np.linspace(min_bound[0], max_bound[0], res)
y_vals = np.linspace(min_bound[1], max_bound[1], res)
z_vals = np.linspace(min_bound[2], max_bound[2], res)
grid_z, grid_y, grid_x = np.meshgrid(z_vals, y_vals, x_vals, indexing='ij')
points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)  # [N, 3]
N = points.shape[0]

# === KNN assign color ===
print("Assigning colors via PyTorch cdist (GPU)...")

# convert to tensor
pc_xyz_torch = torch.tensor(pc_xyz, dtype=torch.float32, device='cuda')   # [N1, 3]
points_torch = torch.tensor(points, dtype=torch.float32, device='cuda')   # [N2, 3]

# compute pairwise distance
batch_size = 3500
grid_rgb = torch.zeros((points_torch.shape[0], 3), device='cuda')

for i in tqdm(range(0, points_torch.shape[0], batch_size), desc="KNN torch"):
    end = min(i + batch_size, points_torch.shape[0])
    batch_points = points_torch[i:end]  # [B, 3]
    
    # 距離矩陣： [B, N1]
    dists = torch.cdist(batch_points, pc_xyz_torch)  
    idxs = torch.argmin(dists, dim=1)  # [B]

    # 查顏色
    batch_rgb = torch.tensor(pc_rgb[idxs.cpu().numpy()], device='cuda')
    grid_rgb[i:end] = batch_rgb

grid_rgb = grid_rgb.cpu().numpy() 

# === Query SDF in 6D ===
print("Querying SDF in 6D...")
batch_size = 4096
sdf_vals = np.zeros(N, dtype=np.float32)
for i in tqdm(range(0, N, batch_size)):
    end = min(i + batch_size, N)
    batch_xyz = points[i:end]    # grid xyz
    batch_rgb = grid_rgb[i:end]  # grid rgb
    batch_6d = np.concatenate([batch_xyz, batch_rgb], axis=1)  # concat. input 6d vector
    with torch.no_grad():
        batch_tensor = torch.tensor(batch_6d, dtype=torch.float32, device=device)
        sdf_vals[i:end] = model(batch_tensor).cpu().numpy().squeeze()

sdf_grid = sdf_vals.reshape(res, res, res)
# sdf_grid[np.abs(sdf_grid) > 0.1] = 1

rgb_grid = grid_rgb.reshape(res, res, res, 3)

# marching cubes前高斯平滑，防破碎
sdf_grid = gaussian_filter(sdf_grid, sigma=1)

# marching cubes
spacing = (z_vals[1] - z_vals[0], y_vals[1] - y_vals[0], x_vals[1] - x_vals[0])
verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
verts = verts + np.array(min_bound)[::-1]
verts = verts[:, [2, 1, 0]]

# mesh 賦色（用KNN找最近grid顏色）
grid_xyz = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
tree2 = cKDTree(grid_xyz)
_, idxs2 = tree2.query(verts)
verts_colors = grid_rgb[idxs2]

mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(verts_colors)
mesh_o3d.compute_vertex_normals()

o3d.io.write_triangle_mesh(output_mesh, mesh_o3d)
print(f"Exported colored mesh: {output_mesh}")

if slice:
    # 設定檔名
    sdf_slice_output_path_x = output_mesh + "_SDFslice_x.png"
    sdf_slice_output_path_y = output_mesh + "_SDFslice_y.png"
    sdf_slice_output_path_z = output_mesh + "_SDFslice_z.png"

    # 擷取三個方向的中間切片
    slice_img_x = sdf_grid[res // 2, :, :]
    slice_img_y = sdf_grid[:, res // 2, :]
    slice_img_z = sdf_grid[:, :, res // 2]

    # 儲存函數（含刻度與 colorbar）
    def save_slice_with_axes(img, path, axis_label):
        plt.figure(figsize=(6, 5))

        vmin, vmax = 0, 1
    
        # 顯示背景色圖
        im = plt.imshow(img, cmap='RdBu', vmin=vmin, vmax=vmax, origin='lower')

        # 加入等高線（contour lines）
        contour_levels = np.linspace(vmin, vmax, 11)  # 共畫11條線
        contours = plt.contour(img, levels=contour_levels, colors='black', linewidths=0.5, origin='lower')
        plt.clabel(contours, fmt="%.2f", fontsize=7)

        # 加入 colorbar 與標題、軸標籤
        cbar = plt.colorbar(im, label='Signed Distance')
        plt.title(f'SDF Slice - {axis_label} axis')
        plt.xlabel('Voxel X' if axis_label != 'X' else 'Voxel Y')
        plt.ylabel('Voxel Y' if axis_label == 'Z' else 'Voxel Z')
        plt.xticks(np.linspace(0, img.shape[1], 5, dtype=int))
        plt.yticks(np.linspace(0, img.shape[0], 5, dtype=int))
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"SDF slice with contour saved to {path}")

    # 儲存三張圖
    save_slice_with_axes(slice_img_x, sdf_slice_output_path_x, 'X')
    save_slice_with_axes(slice_img_y, sdf_slice_output_path_y, 'Y')
    save_slice_with_axes(slice_img_z, sdf_slice_output_path_z, 'Z')

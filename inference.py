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
import faiss
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

from utils import render_pointcloud, str2bool

hist_ = False
slice = False
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
print("Assigning colors via KNN (using faiss GPU, with progress bar)...")
pc_xyz_f32 = np.ascontiguousarray(pc_xyz.astype('float32'))
points_f32 = np.ascontiguousarray(points.astype('float32'))

index = faiss.IndexFlatL2(3)
# if torch.cuda.is_available():
#     print('Using CUDA...')
#     res = faiss.StandardGpuResources()
#     index = faiss.index_cpu_to_gpu(res, 0, index)

batch_size_add = 500_000
n = pc_xyz_f32.shape[0]
for i in tqdm(range(0, n, batch_size_add), desc="Add to faiss index"):
    end = min(i+batch_size_add, n)
    index.add(pc_xyz_f32[i:end])

batch_size_knn = 50000  # 可根據顯存大小調整
grid_rgb = np.zeros((points_f32.shape[0], 3), dtype=np.float32)
for i in tqdm(range(0, points_f32.shape[0], batch_size_knn), desc="KNN color"):
    end = min(i+batch_size_knn, points_f32.shape[0])
    dists, idxs = index.search(points_f32[i:end], 1)
    grid_rgb[i:end] = pc_rgb[idxs.squeeze()]

# === Query SDF (in 6D) ===
print("Querying SDF...")
batch_size = 4096
sdf_vals = np.zeros(N, dtype=np.float32)
for i in tqdm(range(0, N, batch_size)):
    end = min(i + batch_size, N)
    batch_xyz = points[i:end]
    batch_rgb = grid_rgb[i:end]
    batch_6d = np.concatenate([batch_xyz, batch_rgb], axis=1)
    with torch.no_grad():
        batch_tensor = torch.tensor(batch_6d, dtype=torch.float32, device=device)
        sdf_vals[i:end] = model(batch_tensor).cpu().numpy().squeeze()

sdf_grid = sdf_vals.reshape(res, res, res)
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

# ========== 其他分析與可選功能不變 ==========
if slice:
    sdf_slice_output_path = output_mesh + "_SDFslice.png"
    slice_img = sdf_grid[res // 2]
    plt.imsave(sdf_slice_output_path, slice_img, cmap='RdBu', vmin=-1, vmax=1)
    print(f"Raw SDF slice image saved to {sdf_slice_output_path}")


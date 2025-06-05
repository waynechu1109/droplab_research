import argparse
import torch
import numpy as np
from skimage import measure
import open3d as o3d
import json
from model import SDFNet
from tqdm import tqdm

parser = argparse.ArgumentParser(description="SDFNet inference script.")
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
cam_pose = np.array(info["poses"][0])[:3, :4]
R = cam_pose[:, :3]
t = cam_pose[:, 3]
cam_origin = -R.T @ t

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
min_bound = [-1.1, -1.1, -1.1]
max_bound = [1.1, 1.1, 1.1]
x_vals = np.linspace(min_bound[0], max_bound[0], res)
y_vals = np.linspace(min_bound[1], max_bound[1], res)
z_vals = np.linspace(min_bound[2], max_bound[2], res)
grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals)
points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

# === Batched SDF query ===
def batched_query(model, query_np, cam_origin, batch_size=32768):
    out = []
    cam_origin_tensor = torch.tensor(cam_origin, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for i in range(0, len(query_np), batch_size):
            batch = torch.tensor(query_np[i:i+batch_size], dtype=torch.float32).to(device)
            view_dirs = cam_origin_tensor - batch
            view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
            sdf, _ = model(batch, view_dirs=view_dirs)
            out.append(sdf.cpu().numpy())
    return np.concatenate(out, axis=0)

sdf_pred = batched_query(model, points, cam_origin)
sdf_grid = sdf_pred.reshape(res, res, res)

if np.min(sdf_grid) >= 0 or np.max(sdf_grid) <= 0:
    raise ValueError("SDF field has no zero-crossing surface!")

spacing_val = x_vals[1] - x_vals[0]
spacing = (spacing_val, spacing_val, spacing_val)
sdf_grid[np.abs(sdf_grid) > 0.02] = 1
verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
# move verts back to [-1, 1]
verts = verts * spacing_val + min_bound[0] 

verts_tensor = torch.tensor(verts, dtype=torch.float32, device=device)
cam_origin_tensor = torch.tensor(cam_origin, dtype=torch.float32, device=device).unsqueeze(0)
view_dirs = verts_tensor - cam_origin_tensor
view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)

with torch.no_grad():
    _, rgb_pred = model(verts_tensor, view_dirs=view_dirs)
    rgb_np = (rgb_pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(rgb_np.astype(np.float32) / 255.0)
mesh_o3d.compute_vertex_normals()
o3d.io.write_triangle_mesh(output_mesh, mesh_o3d)

print("Mesh exported.")

points_np = np.asarray(pcd.points)
for i in range(3):
    print(f"PointCloud axis={i} bbox:", points_np[:, i].min(), points_np[:, i].max(), 
          f"; Mesh verts axis={i} bbox:", verts[:, i].min(), verts[:, i].max())

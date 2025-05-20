import argparse
import torch
import torch.nn as nn
import numpy as np
from skimage import measure
import open3d as o3d
import trimesh
from model import SDFNet

parser = argparse.ArgumentParser(description="SDFNet inference script.")
parser.add_argument('--res', type=int, default=256, help="Voxel grid resolution.")
parser.add_argument('--ckpt_path', type=str, required=True, help="Path to model checkpoint.")
parser.add_argument('--output_mesh', type=str, required=True, help="Output mesh file path.")
# parser.add_argument('--para', type=float, required=True, help="Parameter want to control.")
parser.add_argument('--file_name', type=str, required=True, help="Pointcloud file name.")
args = parser.parse_args()

res = args.res  # voxel resolution
ckpt_path = args.ckpt_path
output_mesh = args.output_mesh
# para = args.para
file_name = args.file_name

# read pointcloud for the range to construct the mesh
pcd = o3d.io.read_point_cloud(f"data/output_pointcloud_{file_name}_normal.ply")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model checkpoint
try:
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 如果是新格式（包含 pe_freqs）
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        pe_freqs = checkpoint.get("pe_freqs")
        model = SDFNet(pe_freqs=pe_freqs).to(device)
        model.pe_mask = torch.ones(pe_freqs, dtype=torch.bool).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Fail to load model: this is the wrong format")
        exit(1)

    print(f"Model loaded successfully with pe_freqs = {pe_freqs}")

except Exception as e:
    print("Fail to load model:", e)
    exit(1)

# model = SDFNet(pe_freqs=int(para)).to(device)
# model.pe_mask = torch.ones(int(para), dtype=torch.bool).to(device)  # 全開

# # load model
# try:
#     state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
#     model.load_state_dict(state_dict)
# except Exception as e:
#     print("Fail to load model:", e)

# set the model to eval() mode for inference
model.eval()

# set the margin for mesh construction
points = np.asarray(pcd.points)
# compute the bounding box
min_bound = points.min(axis=0)  # [x_min, y_min, z_min]
max_bound = points.max(axis=0)  # [x_max, y_max, z_max]
# add margin (10%)
margin = 0.1 * (max_bound - min_bound)
min_bound -= margin
max_bound += margin

# construct the space for showing mesh
x_vals = np.linspace(min_bound[0], max_bound[0], res)
y_vals = np.linspace(min_bound[1], max_bound[1], res)
z_vals = np.linspace(min_bound[2], max_bound[2], res)
grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals)
points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)  # [M, 3] = [res*res*res, 3]

# same color for every voxel
color_cond = np.array([[0.5, 0.5, 0.5]])
colors = np.repeat(color_cond, len(points), axis=0)
query = np.concatenate([points, colors], axis=1)  # [M, 6]

# batched the pcd then input them to the model
def batched_query(model, query_np, batch_size=32768):
    out = []
    with torch.no_grad():
        for i in range(0, len(query_np), batch_size):
            batch = torch.tensor(query_np[i:i+batch_size], dtype=torch.float32).to(device)
            pred = model(batch).cpu().numpy()
            out.append(pred)
    return np.concatenate(out, axis=0)

sdf_pred = batched_query(model, query)
print("SDF predicted.")

# --- reshape 成 3D grid ---
sdf_grid = sdf_pred.reshape(res, res, res)
print("SDF range:", np.min(sdf_grid), np.max(sdf_grid))

# # --- 只保留最大連通區塊（前處理） ---
# binary_mask = np.abs(sdf_grid) < 0.0008
# labels = measure.label(binary_mask, connectivity=1)
# props = measure.regionprops(labels)

# if len(props) == 0:
#     raise ValueError("找不到任何零交界區域，請檢查 SDF 預測範圍")

# # 找出最大區塊
# largest_region = max(props, key=lambda r: r.area)
# mask = labels == largest_region.label

# # 把非最大區塊的值設成遠離零交界（避免被提取）
# sdf_grid[~mask] = 0.5  # >0 或 <0 均可，只要遠離 level=0 即可

# --- 判斷是否有表面可用 marching cubes ---
if np.min(sdf_grid) >= 0 or np.max(sdf_grid) <= 0:
    raise ValueError("SDF field has no zero-crossing surface!")

# construct mesh using marching cubes
spacing_val = (x_vals[1] - x_vals[0])
spacing = (spacing_val, spacing_val, spacing_val)
verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
# for level in [-0.0005, -0.00025, 0, 0.00025, 0.0005]:
#     verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=level)
verts += np.array([x_vals[0], y_vals[0], z_vals[0]])
verts[:, 0] = -verts[:, 0]

if not np.isfinite(verts).all():
    raise ValueError("Nan/Inf in verts, unable to construct mesh")

# construct mesh
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh.export(output_mesh)
print("Mesh exported to output/sdf_surface.ply")


# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(verts)
# mesh.triangles = o3d.utility.Vector3iVector(faces)
# mesh.compute_vertex_normals()
# o3d.io.write_triangle_mesh("output/sdf_surface.ply", mesh)
# print("Mesh exported to output/sdf_surface.ply")


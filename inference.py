import torch
import torch.nn as nn
import numpy as np
from skimage import measure
import open3d as o3d
import trimesh
from model import SDFNet

res = 256  # voxel resolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SDFNet().to(device)

# load model
try:
    # model.load_state_dict(torch.load("ckpt/sdf_model.pt"))
    # torch.load("ckpt/sdf_model.pt", weights_only=True)
    state_dict = torch.load("ckpt/sdf_model_5000_pointnetPE_n0.01_weight:(20,1,0.5)_v2.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
except Exception as e:
    print("Fail to load model:", e)

model.eval()

# read pointcloud
pcd = o3d.io.read_point_cloud("data/output_pointcloud_1.ply")
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

if not np.isfinite(verts).all():
    raise ValueError("Nan/Inf in verts, unable to construct mesh")

# construct mesh
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
mesh.export("output/sdf_surface_pointnetPE_n0.01_5000_weight:(20,1,0.5)_v2.ply")
print("Mesh exported to output/sdf_surface.ply")


# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(verts)
# mesh.triangles = o3d.utility.Vector3iVector(faces)
# mesh.compute_vertex_normals()
# o3d.io.write_triangle_mesh("output/sdf_surface.ply", mesh)
# print("Mesh exported to output/sdf_surface.ply")


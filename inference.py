import argparse
import torch
import numpy as np
from skimage import measure
import open3d as o3d
import json
from model import SDFNet
from tqdm import tqdm
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from PIL import Image
import glob

from utils import render_pointcloud

hist_ = False
slice = False
render_verts = False

def create_camera_frustum(intrinsic, extrinsic, width, height, scale=0.2):
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()
    near, far = 0.1, scale  # 視距範圍

    # 四個角點 (near plane)
    points = np.array([
        [(0 - cx) * near / fx, (0 - cy) * near / fy, near],
        [(width - cx) * near / fx, (0 - cy) * near / fy, near],
        [(width - cx) * near / fx, (height - cy) * near / fy, near],
        [(0 - cx) * near / fx, (height - cy) * near / fy, near],
        [0, 0, 0]  # camera center
    ])
    
    # 轉換到世界座標系
    points_h = np.concatenate([points, np.ones((5, 1))], axis=1)
    world_pts = (extrinsic @ points_h.T).T[:, :3]

    lines = [
        [4, 0], [4, 1], [4, 2], [4, 3],
        [0, 1], [1, 2], [2, 3], [3, 0]
    ]
    colors = [[1, 0, 0] for _ in lines]  # 紅色

    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(world_pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum

def line_set_to_mesh(line_set, radius=0.002):
    """Convert Open3D LineSet to a mesh of thin cylinders."""
    mesh_list = []
    pts = np.asarray(line_set.points)
    for start_idx, end_idx in np.asarray(line_set.lines):
        start = pts[start_idx]
        end = pts[end_idx]
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(end - start))
        cyl.paint_uniform_color([1, 0, 0])
        cyl.compute_vertex_normals()

        # Align and translate the cylinder
        direction = (end - start)
        direction /= np.linalg.norm(direction)
        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(
            [0, np.arccos(direction[2]), np.arctan2(direction[1], direction[0])]
        )
        cyl.rotate(rot_mat, center=(0, 0, 0))
        cyl.translate(start)
        mesh_list.append(cyl)

    combined = mesh_list[0]
    for mesh in mesh_list[1:]:
        combined += mesh
    return combined

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
focals = np.array(info["focals"])[0][0]
cam_pose = np.array(info["poses"][0])[:3, :4]
R = cam_pose[:, :3]
t = cam_pose[:, 3]
cam_origin = -R.T @ t

# === Load model ===
checkpoint = torch.load(ckpt_path, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    pe_freqs = checkpoint.get("pe_freqs")
    # print(f'pe_freqs: {pe_freqs}')
    view_pe_freqs = checkpoint.get("view_pe_freqs")
    use_tanh_rgb = checkpoint.get("use_tanh_rgb")
    use_feature_vector = checkpoint.get("use_feature_vector")
    feature_vector_size = checkpoint.get("feature_vector_size")
    model = SDFNet(
        pe_freqs=pe_freqs,
        view_pe_freqs=view_pe_freqs,
        use_tanh_rgb=use_tanh_rgb,
        use_feature_vector=use_feature_vector,
        feature_vector_size=feature_vector_size
    ).to(device)
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

points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

# === Batched SDF query ===
def batched_query(model, query_np, cam_origin, batch_size=32768):
    out = []
    cam_origin_tensor = torch.tensor(cam_origin, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for i in range(0, len(query_np), batch_size):
            batch = torch.tensor(query_np[i:i+batch_size], dtype=torch.float32).to(device)
            # view_dirs = cam_origin_tensor - batch
            view_dirs = batch - cam_origin_tensor
            view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
            sdf, _ = model(batch, view_dirs=view_dirs)
            out.append(sdf.cpu().numpy())
    return np.concatenate(out, axis=0)

sdf_pred = batched_query(model, points, cam_origin)

# 排除極端值後，找最大頻率落點（histogram mode）
hist, bin_edges = np.histogram(sdf_pred, bins=200, range=(-3, 3))
mode_center = bin_edges[np.argmax(hist)]

# 對齊該模式中心
sdf_pred_centered = sdf_pred - mode_center
print("New median after centering:", np.median(sdf_pred_centered))

sdf_grid = sdf_pred_centered.reshape(res, res, res)
# sdf_grid = sdf_pred.reshape(res, res, res)

if np.min(sdf_grid) >= 0 or np.max(sdf_grid) <= 0:
    raise ValueError("SDF field has no zero-crossing surface!")

spacing_val = x_vals[1] - x_vals[0]
spacing = (z_vals[1] - z_vals[0], y_vals[1] - y_vals[0], x_vals[1] - x_vals[0])

sdf_grid[np.abs(sdf_grid) > 0.2] = 1
# thresh = np.percentile(np.abs(sdf_pred), 99.9999999)  # 只排除極端值
# sdf_grid[np.abs(sdf_grid) > thresh] = 1
# sdf_grid = np.clip(sdf_grid, -3.0, 3.0)

verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
# move verts back to [-1, 1]
verts = verts + np.array(min_bound)[::-1]
verts = verts[:, [2, 1, 0]]

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

print("SDF value stats:")
print("Min:", np.min(sdf_pred))
print("Max:", np.max(sdf_pred))
print("Mean:", np.mean(sdf_pred))
print("Percentiles:", np.percentile(sdf_pred, [0.1, 1, 25, 50, 75, 99, 99.9]))

print("SDF grid min/max:", np.min(sdf_grid), np.max(sdf_grid))
print("SDF grid shape:", sdf_grid.shape)
print("Zero-crossing count:", np.sum(np.abs(sdf_grid) < 0.01))

print("First few verts:", verts[:5])
print("Point cloud bounds:", np.asarray(pcd.points).min(0), np.asarray(pcd.points).max(0))
print(f"Shape of verts: {verts.shape}")
print("Verts bounds:", verts.min(0), verts.max(0))

points_np = np.asarray(pcd.points)
for i in range(3):
    print(f"PointCloud axis={i} bbox:", points_np[:, i].min(), points_np[:, i].max(), 
          f"; Mesh verts axis={i} bbox:", verts[:, i].min(), verts[:, i].max())
    
if slice:
    sdf_slice_output_path = output_mesh + "_SDFslice.png"
    slice_img = sdf_grid[res // 2]
    plt.imsave(sdf_slice_output_path, slice_img, cmap='RdBu', vmin=-1, vmax=1)
    print(f"Raw SDF slice image saved to {sdf_slice_output_path}")

if hist_:
    plt.hist(sdf_pred, bins=200, range=(-3, 3))
    plt.axvline(np.median(sdf_pred), color='r', label='median')
    plt.axvline(mode_center, color='g', label='mode center')
    plt.axvline(0, color='k', linestyle='--', label='zero')
    plt.legend()
    plt.title("SDF value distribution")
    plt.savefig(output_mesh + "_sdf_hist.png")


if render_verts:
    # Step 0: intrinsic
    image_candidates = glob.glob(f"data/{file_name}.jpg") + glob.glob(f"data/{file_name}.png")
    if len(image_candidates) == 0:
        raise FileNotFoundError(f"No image found for {file_name}.jpg or .png in data/")
    print("Successfully read input image.")
    img_path = image_candidates[0] 
    with Image.open(img_path) as img:
        W, H = img.size
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=W, height=H, fx=focals, fy=focals, cx=W / 2, cy=H / 2)

    # Step 1-4: normalize camera pose
    R = cam_pose[:3, :3]
    t = cam_pose[:3, 3]
    cam_pos = -R.T @ t
    cam_pos_norm = (cam_pos - centre) / scale
    t_norm = -R @ cam_pos_norm

    cam_pose_normed = np.eye(4)
    cam_pose_normed[:3, :3] = R
    cam_pose_normed[:3, 3] = t_norm

    # Step 5: assign to Open3D param
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = intrinsic
    param.extrinsic = cam_pose_normed

    # === 渲染 ===
    mesh_points = torch.tensor(np.asarray(mesh_o3d.vertices), dtype=torch.float32, device=device)
    mesh_colors = torch.tensor(np.asarray(mesh_o3d.vertex_colors), dtype=torch.float32, device=device)

    rendered, _ = render_pointcloud(
        mesh_points, 
        mesh_colors, 
        torch.tensor(cam_pose_normed, dtype=torch.float32, device=device),
        torch.tensor(intrinsic.intrinsic_matrix, dtype=torch.float32, device=device),
        (H, W)
    )

    rendered_np = rendered.cpu().numpy()
    plt.imsave(output_mesh + "_rendered_point.png", rendered_np)
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
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

from utils import render_pointcloud, str2bool

hist_ = False
slice = False
render_verts = False

def find_zero_crossing_rgb(model, xyz, rgb_candidates, device, threshold=0.01):
    # xyz: (N, 3)
    # rgb_candidates: (K, 3), 建議可設定如np.linspace(0, 1, 16)三軸組合
    N = xyz.shape[0]
    K = rgb_candidates.shape[0]
    xyz_tiled = np.repeat(xyz, K, axis=0)  # [N*K, 3]
    rgb_tiled = np.tile(rgb_candidates, (N, 1))  # [N*K, 3]
    query_6d = np.concatenate([xyz_tiled, rgb_tiled], axis=1)
    with torch.no_grad():
        batch = torch.tensor(query_6d, dtype=torch.float32, device=device)
        sdf_vals = model(batch)
        sdf_vals = sdf_vals.cpu().numpy().reshape(N, K)
        # 對每個xyz, 找到最接近0的rgb
        best_idx = np.argmin(np.abs(sdf_vals), axis=1)
        best_rgb = rgb_candidates[best_idx]
        best_sdf = sdf_vals[np.arange(N), best_idx]
    # 過濾掉sdf太遠的點
    mask = np.abs(best_sdf) < threshold
    return xyz[mask], best_rgb[mask]

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
parser.add_argument('--is_a100', type=str2bool, required=True, help="Training on A100 or not.")
args = parser.parse_args()

res = args.res
ckpt_path = args.ckpt_path
output_mesh = args.output_mesh
file_name = args.file_name
is_a100 = args.is_a100

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
    model = SDFNet(
        pe_freqs=pe_freqs
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

# === 1. marching cubes產生verts ===
# 建立 3D 空間格點
points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)  # [N, 3]
N = points.shape[0]

# 建立顏色搜尋表
if is_a100:
    rgb_lin = np.linspace(0, 1, 16)  # 視顏色解析度調整
else:
    rgb_lin = np.linspace(0, 1, 8)  # 視顏色解析度調整

rgb_candidates = np.stack(np.meshgrid(rgb_lin, rgb_lin, rgb_lin, indexing='ij'), axis=-1).reshape(-1, 3)  # [K, 3]
K = rgb_candidates.shape[0]

batch_size = 256
min_sdf = np.zeros(N, dtype=np.float32)
best_rgb = np.zeros((N, 3), dtype=np.float32)

print("Querying 6D SDF for marching cubes grid ...")
for i in tqdm(range(0, N, batch_size)):
    end = min(i+batch_size, N)
    xyz_batch = points[i:end]  # [B, 3]
    B = xyz_batch.shape[0]
    # 擴展為 B*K 個6D點
    xyz_tile = np.repeat(xyz_batch, K, axis=0)  # [B*K, 3]
    rgb_tile = np.tile(rgb_candidates, (B, 1))  # [B*K, 3]
    query_6d = np.concatenate([xyz_tile, rgb_tile], axis=1)  # [B*K, 6]
    with torch.no_grad():
        batch_6d = torch.tensor(query_6d, dtype=torch.float32, device=device)
        sdf_vals = model(batch_6d).cpu().numpy().reshape(B, K)
    # min_indices = np.argmin(np.abs(sdf_vals), axis=1)
    # min_sdf[i:end] = sdf_vals[np.arange(B), min_indices]
    # best_rgb[i:end] = rgb_candidates[min_indices]
    topk = 4
    sort_idxs = np.argsort(np.abs(sdf_vals), axis=1)
    topk_idxs = sort_idxs[:, :topk]
    rgb_avg = np.mean(rgb_candidates[topk_idxs], axis=1)
    min_sdf[i:end] = np.mean(sdf_vals[np.arange(B)[:, None], topk_idxs], axis=1)
    best_rgb[i:end] = rgb_avg


sdf_grid = min_sdf.reshape(res, res, res)
rgb_grid = best_rgb.reshape(res, res, res, 3)

sdf_grid[np.abs(sdf_grid) > 0.05] = 1
sdf_grid = gaussian_filter(sdf_grid, sigma=1)

spacing = (z_vals[1] - z_vals[0], y_vals[1] - y_vals[0], x_vals[1] - x_vals[0])
verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
verts = verts + np.array(min_bound)[::-1]
verts = verts[:, [2, 1, 0]]

# 對每個mesh vertex找最靠近的格點顏色
grid_xyz = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
tree = cKDTree(grid_xyz)
dists, idxs = tree.query(verts)
verts_colors = best_rgb[idxs]

# 2. 組成 open3d mesh
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(verts_colors)
mesh_o3d.compute_vertex_normals()

# 3. 輸出
o3d.io.write_triangle_mesh(output_mesh, mesh_o3d)
print(f"Exported colored mesh: {output_mesh}")


# print("Mesh exported.")

# print("SDF value stats:")
# print("Min:", np.min(sdf_pred))
# print("Max:", np.max(sdf_pred))
# print("Mean:", np.mean(sdf_pred))
# print("Percentiles:", np.percentile(sdf_pred, [0.1, 1, 25, 50, 75, 99, 99.9]))

# print("SDF grid min/max:", np.min(sdf_grid), np.max(sdf_grid))
# print("SDF grid shape:", sdf_grid.shape)
# print("Zero-crossing count:", np.sum(np.abs(sdf_grid) < 0.01))

# print("First few verts:", verts[:5])
# print("Point cloud bounds:", np.asarray(pcd.points).min(0), np.asarray(pcd.points).max(0))
# print(f"Shape of verts: {verts.shape}")
# print("Verts bounds:", verts.min(0), verts.max(0))

# points_np = np.asarray(pcd.points)
# for i in range(3):
#     print(f"PointCloud axis={i} bbox:", points_np[:, i].min(), points_np[:, i].max(), 
#           f"; Mesh verts axis={i} bbox:", verts[:, i].min(), verts[:, i].max())
    
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
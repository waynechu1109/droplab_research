import open3d as o3d
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def render_pointcloud(points: torch.Tensor,
                      colors: torch.Tensor,
                      cam_pose: torch.Tensor,
                      K: torch.Tensor,
                      image_size: tuple[int, int]
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    device = points.device
    H, W = image_size

    # 1) Transform to camera coords
    n = points.shape[0]
    pts_h = torch.cat([points, torch.ones(n, 1, device=device)], dim=1)  # (n,4)
    cam_h = (cam_pose @ pts_h.T).T                                       # (n,4)
    Xc, Yc, Zc = cam_h[:, 0], cam_h[:, 1], cam_h[:, 2]

    # 2) Project and normalize
    proj = (K @ cam_h[:, :3].T).T                                        # (n,3)
    u = (proj[:, 0] / proj[:, 2]).round().long()
    v = (proj[:, 1] / proj[:, 2]).round().long()

    # 3) Keep only visible points
    valid = (Zc > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Zc, cols = u[valid], v[valid], Zc[valid], colors[valid]

    # 4) Buffers
    image = torch.zeros((H, W, 3), device=device)
    depth = torch.full((H, W), float('inf'), device=device)

    # 5) Depth test
    for xi, yi, zi, ci in zip(u, v, Zc, cols):
        if zi < depth[yi, xi]:
            depth[yi, xi] = zi
            image[yi, xi] = ci

    return image, depth

# --- example usage ---
if __name__ == "__main__":
    # 1. read pointcloud
    # pcd = o3d.io.read_point_cloud("data/output_pointcloud_all.ply")
    pcd = o3d.io.read_point_cloud("data/output_pointcloud_0.ply")
    # pcd = o3d.io.read_point_cloud("data/output_pointcloud_1.ply")
    points_np = np.asarray(pcd.points)    # (n,3)
    colors_np = np.asarray(pcd.colors)    # (n,3) in [0,1]

    # 2. convert to torch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.from_numpy(points_np).float().to(device)
    colors = torch.from_numpy(colors_np).float().to(device)

    # 3. Camera parameters
    H, W = 480, 640
    fx = fy = 880.6520 # focal from dust3r 
    # fx = fy = 873.5108 # focal from dust3r
    cx, cy = W / 2, H / 2

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32, device=device)

    cam_pose_np = np.array([
        [ 0.8722, -0.4330,  0.2276, -0.0614],
        [ 0.4497,  0.8928, -0.0248,  0.0047],
        [-0.1925,  0.1240,  0.9734,  0.0042],
        [ 0.0000,  0.0000,  0.0000,  1.0000],  # pose 0
    ], dtype=np.float32)

    # cam_pose_np = np.array([
    #     [ 0.9999,  0.0036, -0.0116,  0.0000],
    #     [-0.0037,  1.0000, -0.0041,  0.0000],
    #     [ 0.0115,  0.0041,  0.9999,  0.0000],   # pose 1
    #     [ 0.0000,  0.0000,  0.0000,  1.0000]
    # ], dtype=np.float32)

    cam_pose = torch.from_numpy(cam_pose_np).to(device)
    # cam_pose = torch.inverse(cam_pose) 

    # 5. render
    img, dep = render_pointcloud(points, colors, cam_pose, K, (H, W))

    # 6. show result
    img_np = img.cpu().numpy()
    dep_np = dep.cpu().numpy()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("RGB Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Depth Map")
    plt.imshow(dep_np, cmap='plasma')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

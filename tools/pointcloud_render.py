import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from PIL import Image
import glob

from utils import render_pointcloud

parser = argparse.ArgumentParser(description="Point Cloud rendering script.")
parser.add_argument('--file', type=str, help="File name to render.")
args = parser.parse_args()
file_name = args.file

if __name__ == "__main__":
    # 1. Read point cloud
    pcd = o3d.io.read_point_cloud(f"../data/output_pointcloud_{file_name}_normal.ply")
    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors)

    # 2. Convert to torch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.from_numpy(points_np).float().to(device)
    colors = torch.from_numpy(colors_np).float().to(device)

    # 3. Load intrinsic & poses from JSON
    with open(f"../data/pointcloud_{file_name}_normal_info.json", "r") as f:
        info = json.load(f)

    focals = info["focals"]  # shape: [N, 1]
    poses = info["poses"]    # shape: [N, 4, 4]
    centre = torch.tensor(info["centre"], dtype=torch.float32, device=device)
    scale = torch.tensor(info["scale"], dtype=torch.float32, device=device)

    # unnormalize pcd
    points = points * scale + centre

    # 讀取圖片尺寸
    image_candidates = glob.glob(f"../data/{file_name}.jpg") + glob.glob(f"../data/{file_name}.png")
    if len(image_candidates) == 0:
        raise FileNotFoundError(f"No image found for {file_name}.jpg or .png in data/")
    print("Successfully read input image.")
    img_path = image_candidates[0] 
    with Image.open(img_path) as img:
        W, H = img.size

    cx, cy = W / 2, H / 2

    # 4. Loop over each pose
    for i, (f, pose) in enumerate(zip(focals, poses)):
        fx = fy = f[0]  # assume square pixels
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=torch.float32, device=device)

        cam_pose = torch.tensor(pose, dtype=torch.float32, device=device)

        img, dep = render_pointcloud(points, colors, cam_pose, K, (H, W))

        # Convert to numpy for visualization
        img_np = img.cpu().numpy()
        dep_np = dep.cpu().numpy()

        plt.figure(figsize=(12, 5))
        plt.suptitle(f"Rendered View {file_name}_{i}", fontsize=14)

        plt.subplot(1, 2, 1)
        plt.title("RGB Image")
        plt.imshow(img_np)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Depth Map")
        depth_plot = plt.imshow(dep_np, cmap='plasma')
        plt.axis("off")
        cbar = plt.colorbar(depth_plot, fraction=0.046, pad=0.04)
        cbar.set_label("Depth", rotation=270, labelpad=15)

        plt.tight_layout()
        # plt.show()

        plt.savefig(f"../render_result/rendered_view_{file_name}_{i}.png")

        print(f"[{i+1}/{len(focals)}] Finish rendering.")

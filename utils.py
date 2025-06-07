import torch

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
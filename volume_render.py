import torch
import torch.nn.functional as F

def sdf_to_sigma(sdf, alpha=10.0, beta=0.1):
    # Convert SDF to density (sigma), ref: NeuS paper
    # alpha: sharpness, beta: defines narrow band
    return alpha * torch.exp(- (sdf ** 2) / beta)

def project_to_image(points, K, pose, image, H, W):
    """
    Projects 3D points (world space) into 2D image pixels and samples RGB values.

    Args:
        points: [N, 3] - world space 3D points
        K: [3,3] - camera intrinsics
        pose: [4,4] - world-to-camera transform
        image: [H, W, 3] - input image
        H, W: image height and width

    Returns:
        sampled_rgb: [N, 3]
    """
    device = points.device

    # transform points to camera space
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_cam = (R @ points.T + t.unsqueeze(1)).T  # [N, 3]

    # project to pixel coordinates
    proj = (K @ points_cam.T).T  # [N, 3]
    xy = proj[:, :2] / (proj[:, 2:3] + 1e-8)  # [N, 2]

    # normalize to [-1, 1] for grid_sample
    x_norm = xy.clone()
    x_norm[:, 0] = (x_norm[:, 0] / (W - 1)) * 2 - 1
    x_norm[:, 1] = (x_norm[:, 1] / (H - 1)) * 2 - 1
    grid = x_norm.view(1, -1, 1, 2)  # [1, N, 1, 2]

    # sample RGB from image
    image = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    sampled_rgb = F.grid_sample(image, grid, align_corners=True, mode='bilinear', padding_mode='border')  # [1, 3, N, 1]
    sampled_rgb = sampled_rgb.squeeze().T  # [N, 3]

    # remove invalid projections (z <= 0 or outside view)
    z = points_cam[:, 2]
    valid = (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)

    sampled_rgb[~valid] = 0.5  # default gray for invalid points

    return sampled_rgb


def volume_rendering(model, rays_o, rays_d, K, pose, image, image_size, num_samples=32, near=0.1, far=1.5, device='cuda'):  
    H, W = image_size
    R = rays_o.shape[0]  # R: number of rays

    # Step 1: Sample points along each ray
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # [N]
    z_vals = z_vals.expand(R, num_samples)  # [R, N]
    points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # [R, N, 3]
    points = points * model.scale + model.centre

    # Step 2: Evaluate SDF at points
    points_flat = points.reshape(-1, 3)  # [R*N, 3]

    # 實際從 image 取 RGB
    rgb_vals = project_to_image(points_flat, K, pose, image, H, W)  # [R*N, 3]
    rgb_vals = rgb_vals.reshape(R, num_samples, 3)

    model_input = torch.cat([points_flat, rgb_vals.reshape(-1, 3)], dim=-1)  # [R*N, 6]
    sdf = model(model_input).reshape(R, num_samples)  # [R, N]

    # Step 3: SDF to density
    sigma = sdf_to_sigma(sdf)

    # Step 4: Compute weights
    delta = z_vals[:, 1:] - z_vals[:, :-1]
    delta = torch.cat([delta, 1e10 * torch.ones_like(delta[:, :1])], dim=-1)
    sigma = torch.clamp(sigma, min=0.0, max=1e3)  # 避免極端值爆炸
    delta = torch.clamp(delta, min=1e-5)          # 避免除以 0

    alpha = 1.0 - torch.exp(-sigma * delta)
    alpha = torch.clamp(alpha, 0.0, 1.0)
    T = torch.cumprod(torch.cat([torch.ones((R, 1), device=device), 1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
    weights = T * alpha  # [R, N]
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 5: Composite final color
    rendered_color = torch.sum(weights.unsqueeze(-1) * rgb_vals, dim=1)  # [R, 3]
    rendered_color = torch.nan_to_num(rendered_color, nan=0.0, posinf=1.0, neginf=0.0)

    return rendered_color

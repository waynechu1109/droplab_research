import torch
import torch.nn.functional as F

def sdf_to_sigma(sdf, alpha=75.0, beta=0.03):
    # Convert SDF to density (sigma), ref: NeuS paper
    # alpha: sharpness, beta: defines narrow band
    return alpha * torch.exp(- (sdf ** 2) / beta)

def volume_rendering(
        model, rays_o, rays_d, 
        view_dirs=None, K=None, 
        pose=None, image=None, 
        image_size=(256, 256), 
        num_samples=32, near=0.1, 
        far=1.5, device='cuda'
    ):
    H, W = image_size
    R = rays_o.shape[0]  # R: number of rays

    # Step 1: Sample points along each ray
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # [N]
    z_vals = z_vals.expand(R, num_samples)  # [R, N]
    points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # [R, N, 3]
    points = points * model.scale + model.centre
    points_flat = points.reshape(-1, 3)  # [R*N, 3]

    # Step 2: View directions per sample point
    if view_dirs is not None:
        view_dirs = F.normalize(view_dirs, dim=-1)        # [R, 3]
        view_dirs_flat = view_dirs.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 3)  # [R*N, 3]
    else:
        view_dirs_flat = None

    # Step 3: Input to model (points, view dirs)
    if view_dirs_flat is not None:
        sdf, pred_rgb = model(points_flat, view_dirs_flat, return_rgb=True)
        sdf = sdf.detach()  # 不讓 sdf 回傳梯度，但保留 pred_rgb 路徑
    else:
        sdf, pred_rgb = model(points_flat)

    sdf = sdf.reshape(R, num_samples)
    pred_rgb = pred_rgb.reshape(R, num_samples, 3)

    # Step 4: SDF to density
    sigma = sdf_to_sigma(sdf)

    # Step 5: Compute weights
    with torch.no_grad():
        delta = z_vals[:, 1:] - z_vals[:, :-1]
        delta = torch.cat([delta, 1e10 * torch.ones_like(delta[:, :1])], dim=-1)
        sigma = torch.clamp(sigma, min=0.0, max=1e3)  # 避免極端值爆炸
        delta = torch.clamp(delta, min=1e-5)          # 避免除以 0

        alpha = 1.0 - torch.exp(-sigma * delta)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        T = torch.cumprod(torch.cat([torch.ones((R, 1), device=device), 1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
        weights = T * alpha  # [R, N]
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 6: Composite final color
    rendered_color = torch.sum(weights.unsqueeze(-1) * pred_rgb, dim=1)  # [R, 3]
    rendered_color = torch.nan_to_num(rendered_color, nan=0.0, posinf=1.0, neginf=0.0)

    # Composite points (optional)
    ray_pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # [R, N, 3]
    composite_pts = torch.sum(weights.unsqueeze(-1) * ray_pts, dim=1)  # [R, 3]
    composite_pts = torch.nan_to_num(composite_pts, nan=0.0, posinf=0.0, neginf=0.0)

    return rendered_color, composite_pts

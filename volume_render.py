import torch
import torch.nn.functional as F

def sdf_to_sigma(sdf, alpha=10.0, beta=0.1):
    # Convert SDF to density (sigma), ref: NeuS paper
    # alpha: sharpness, beta: defines narrow band
    return alpha * torch.exp(- (sdf ** 2) / beta)

def volume_rendering(model, rays_o, rays_d, num_samples=64, near=0.1, far=1.5, device='cuda'):
    R = rays_o.shape[0]  # R: number of rays

    # Step 1: Sample depth t along each ray
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # [N_samples]
    z_vals = z_vals.expand(R, num_samples)  # [R, N]

    # [R, N, 3] = [R, 1, 3] + [R, 1, 3] * [R, N, 1]
    points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)  # [R, N, 3]

    # Step 2: Query model: [R*N, 3+3]
    points_flat = points.reshape(-1, 3)

    dummy_rgb = torch.ones_like(points_flat) * 0.5  # 使用 dummy RGB（因為 model expects 6D）
    
    model_input = torch.cat([points_flat, dummy_rgb], dim=-1)  # [R*N, 6]
    sdf = model(model_input).reshape(R, num_samples)  # [R, N]

    # Step 3: Convert SDF → density (σ)
    sigma = sdf_to_sigma(sdf)  # [R, N]

    # Step 4: Compute alpha and weights
    delta = z_vals[:, 1:] - z_vals[:, :-1]  # [R, N-1]
    delta = torch.cat([delta, 1e10 * torch.ones_like(delta[:, :1])], dim=-1)  # pad last interval
    alpha = 1.0 - torch.exp(-sigma * delta)  # [R, N]
    T = torch.cumprod(torch.cat([torch.ones((R, 1), device=device), 1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]  # [R, N]
    weights = T * alpha  # [R, N]

    # Step 5: Assign color to each point（暫時使用 dummy color）
    rgb_vals = torch.ones((R, num_samples, 3), device=device) * 0.5  # 假設每個點顏色都是灰色 [0.5, 0.5, 0.5]
    # 若有 learned RGB 預測，可從 model 取出 feature → RGB decoder

    # Step 6: Final composite color
    rendered_color = torch.sum(weights.unsqueeze(-1) * rgb_vals, dim=1)  # [R, 3]

    return rendered_color  # 每條 ray 的 RGB 顏色

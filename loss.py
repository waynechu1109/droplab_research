import torch
import torch.nn.functional as F
from volume_render import volume_rendering

def generate_rays(H, W, K, pose, device):
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    dirs = torch.stack([(i - K[0,2]) / K[0,0], (j - K[1,2]) / K[1,1], torch.ones_like(i)], -1)  # [H, W, 3]
    dirs = dirs.reshape(-1, 3)
    rays_d = (pose[:3, :3] @ dirs.T).T  # [N, 3]
    rays_o = pose[:3, 3].expand_as(rays_d)  # [N, 3]
    return rays_o.to(device), rays_d.to(device)

# refer to NeuS
def compute_render_color_loss(model, rays_o, rays_d, gt_image, cam_pose, K, image_size):
    device = rays_o.device
    H, W = image_size

    # 反向方向
    view_dirs = -F.normalize(rays_d, dim=-1)  # [R, 3]

    # rendering（含 sample 點）
    rendered_color, composite_pts = volume_rendering(model, rays_o, rays_d, view_dirs=view_dirs, K=K, pose=cam_pose, image=gt_image, image_size=(H, W), device=device)

    # 投影到像素座標
    R, t = cam_pose[:3, :3], cam_pose[:3, 3]
    x_cam = (R @ composite_pts.T + t.unsqueeze(1)).T  # [N, 3]
    proj = (K @ x_cam.T).T
    x_pixel = proj[:, :2] / (proj[:, 2:3] + 1e-8)  # 防止除以0
    x_pixel = x_pixel.round().long()

    u, v = x_pixel[:, 0], x_pixel[:, 1]

    # 有效像素
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if valid.sum() == 0:
        print("WARNING: No valid projected pixels for render loss.")
        return torch.tensor(0.0, device=device)

    u, v = u[valid], v[valid]
    rendered_color = rendered_color[valid].clamp(0, 1)  # 限制範圍
    gt_rgb = gt_image[v, u]  # [N, 3]

    # render confidence mask（同樣是 [N, 1]）
    render_conf = (gt_rgb.mean(dim=-1, keepdim=True) > 0.2).float()

    # L1 loss + mask
    # loss_render = F.l1_loss(rendered_color * render_conf, gt_rgb * render_conf)
    loss_render = F.mse_loss(rendered_color * render_conf, gt_rgb * render_conf)


    return loss_render


# refer to SparseNeuS
def compute_sparse_loss(model, x, num_samples=10000, box_margin=0.1, tau=20.0):
    with torch.no_grad():
        min_bound = x[:, :3].min(dim=0)[0]
        max_bound = x[:, :3].max(dim=0)[0]
        margin = (max_bound - min_bound) * box_margin
        min_bound -= margin
        max_bound += margin

        # Uniform sampling in expanded bounding box
        uniform_xyz = torch.rand((num_samples, 3), device=x.device) * (max_bound - min_bound) + min_bound
        # uniform_rgb = torch.ones_like(uniform_xyz) * 0.5
        # uniform_input = torch.cat([uniform_xyz, uniform_rgb], dim=-1)

    # Predict SDF values
    sdf_pred, _ = model(uniform_xyz, return_rgb=False)

    # Sparse loss: exp(-tau * |s(q)|)
    sparse_loss = torch.exp(-tau * sdf_pred.abs()).mean()

    return sparse_loss

def compute_color_consistency_loss(x, f_x, alpha=10.0, sample_size=1024):
    N = x.shape[0]
    idx = torch.randperm(N)[:sample_size]
    x_sub = x[idx]  # [M,6]
    f_sub = f_x[idx]  # [M]
    rgb = x_sub[:, 3:]  # [M,3]

    diff = rgb.unsqueeze(1) - rgb.unsqueeze(0)  # [M,M,3]
    color_dist2 = (diff ** 2).sum(-1)  # [M,M]
    w = torch.exp(-alpha * color_dist2)

    sdf_diff2 = (f_sub.unsqueeze(1) - f_sub.unsqueeze(0)) ** 2  # [M,M]
    loss = (w * sdf_diff2).mean()
    return 100 * loss

def compute_normal_loss(model, x, normals, batch_size=8192):
    N = x.shape[0]
    total_loss = 0.0
    device = x.device

    for i in range(0, N, batch_size):
        x_batch = x[i:i+batch_size].clone().detach().requires_grad_(True)
        normals_batch = normals[i:i+batch_size]

        f_pred, _ = model(x_batch, return_rgb=False)

        grads = torch.autograd.grad(
            outputs=f_pred,
            inputs=x_batch,
            grad_outputs=torch.ones_like(f_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, :3]

        # only calculate normal loss near the surfaces
        mask = (f_pred.abs() < 0.05)
        if mask.any():
            grads = grads[mask]
            normals_batch = normals_batch[mask]

        # cos_sim = F.cosine_similarity(grads, normals_batch, dim=1)
        grads = F.normalize(grads, dim=1)
        normals_batch = F.normalize(normals_batch, dim=1)
        cos_sim = F.cosine_similarity(grads, normals_batch, dim=1)
        loss = ((1 - cos_sim) ** 2).mean()
        total_loss += loss * len(x_batch)

    return total_loss / N

# total loss calculation
def compute_loss(model, x, x_noisy_full, epsilon, normals, H, W, K, cam_pose, gt_image, is_a100):
    x.requires_grad_(True)
    f_x, _ = model(x, return_rgb=False)
    f_x_noisy, _ = model(x_noisy_full, return_rgb=False)

    # Part 1: MSE between predicted SDF at xⁿ and actual |ε|
    # true_dist = epsilon.norm(dim=1)
    # loss_sdf = F.mse_loss(f_x_noisy, true_dist)
    normals = torch.tensor(normals, dtype=torch.float32, device=x.device)
    signed_dist = torch.sum(epsilon * normals, dim=1)
    loss_sdf = F.mse_loss(f_x_noisy, signed_dist)

    # Part 2: Zero loss: ||f(x, c)||
    loss_zero = f_x.abs().mean()

    # Part 3: Eikonal: ||∇f(x^n, c) - 1||
    x_noisy_full.requires_grad_(True) # enable partial differentiation
    # x.requires_grad_()
    f_pred, _ = model(x_noisy_full, return_rgb=False)
    grads = torch.autograd.grad(
        outputs=f_pred,       # f(xⁿ, c)
        # inputs=x,  # input with 6 dimensions: [x, y, z, r, g, b]
        inputs=x_noisy_full,
        grad_outputs=torch.ones_like(f_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, :3]  # take the gradieant w.r.t. (x, y, z)

    grad_norm = torch.norm(grads, dim=-1)

    # narrow band mask: 只在靠近表面的位置計算 eikonal loss
    sdf_abs = torch.abs(f_pred.detach().squeeze(-1))  # detach 避免回傳梯度
    mask = sdf_abs < 0.2  # threshold 可調整：0.05～0.2 之間都可嘗試
    if mask.any():
        loss_eikonal = ((grad_norm[mask] - 1) ** 2).mean()
    else:
        loss_eikonal = torch.tensor(0.0, device=f_pred.device)

    # Part 5: Normal loss
    if not isinstance(normals, torch.Tensor):
        normals = torch.tensor(normals, dtype=torch.float32, device=x.device)
    else:
        normals = normals.clone().detach().to(dtype=torch.float32, device=x.device)
    loss_normal = compute_normal_loss(model, x, normals, batch_size=50000)

    # Part 6: Color Consistency loss
    # loss_consistency = compute_color_consistency_loss(x, f_x, alpha=20.0, sample_size=2048)

    # Part 7: Sparse loss
    if is_a100:
        loss_sparse = compute_sparse_loss(model, x, num_samples=100000)
    else:
        loss_sparse = compute_sparse_loss(model, x, num_samples=7500)

    # Part 8: Color render loss
    rays_o, rays_d = generate_rays(H, W, K, cam_pose, x.device)
    # 隨機取 subset
    if is_a100:
        max_rays = 8192
    else:
        max_rays = 256
    perm = torch.randperm(rays_o.shape[0], device=rays_o.device)[:max_rays]
    rays_o, rays_d = rays_o[perm], rays_d[perm]
    loss_render = compute_render_color_loss(model, rays_o, rays_d, gt_image, cam_pose, K, (H, W))
    
    return loss_sdf, loss_zero, loss_eikonal, loss_normal, loss_sparse, loss_render
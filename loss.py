import torch
import torch.nn.functional as F
from volume_render import volume_rendering
import matplotlib.pyplot as plt

# def generate_rays(H, W, K, pose, device):
#     # build pixels coordinates
#     i, j = torch.meshgrid(
#         torch.arange(W, device=device),
#         torch.arange(H, device=device),
#         indexing='xy'
#     )
#     dirs = torch.stack([(i - K[0,2]) / K[0,0], (j - K[1,2]) / K[1,1], torch.ones_like(i)], -1)  # [H, W, 3]
#     dirs = dirs.reshape(-1, 3)
#     rays_d = (pose[:3, :3] @ dirs.T).T  # [H*W, 3]     # rays' directions in world coordinates
#     rays_o = pose[:3, 3].expand_as(rays_d)  # [H*W, 3] # camera center in world coordinates
#     return rays_o.to(device), rays_d.to(device)

# refer to NeuS
# def compute_render_color_loss(model, rays_o, rays_d, gt_image, cam_pose, K, image_size):
#     device = rays_o.device
#     H, W = image_size

#     # 反向方向
#     view_dirs = -F.normalize(rays_d, dim=-1)  # [R, 3]

#     # rendering（含 sample 點）
#     rendered_color, composite_pts = volume_rendering(model, rays_o, rays_d, view_dirs=view_dirs, K=K, pose=cam_pose, image=gt_image, image_size=(H, W), device=device)

#     # 投影到像素座標
#     R, t = cam_pose[:3, :3], cam_pose[:3, 3]
#     x_cam = (R @ composite_pts.T + t.unsqueeze(1)).T  # [N, 3]
#     proj = (K @ x_cam.T).T
#     x_pixel = proj[:, :2] / (proj[:, 2:3] + 1e-8)  # 防止除以0
#     x_pixel = x_pixel.round().long()

#     u, v = x_pixel[:, 0], x_pixel[:, 1]

#     # 有效像素
#     valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#     if valid.sum() == 0:
#         print("WARNING: No valid projected pixels for render loss.")
#         return torch.tensor(0.0, device=device)

#     u, v = u[valid], v[valid]
#     rendered_color = rendered_color[valid].clamp(0, 1)  # 限制範圍
#     gt_rgb = gt_image[v, u]  # [number of rays, 3]
#     # print(f"gt_rgb.shape: {gt_rgb.shape}")

#     # render confidence mask（同樣是 [N, 1]）
#     # render_conf = (gt_rgb.mean(dim=-1, keepdim=True) > 0.2).float()

#     # mask
#     # loss_render = F.l1_loss(rendered_color * render_conf, gt_rgb * render_conf)
#     # loss_render = F.mse_loss(rendered_color * render_conf, gt_rgb * render_conf)
#     # loss_render = F.mse_loss(rendered_color, gt_rgb)

#     loss_render = torch.abs(rendered_color - gt_rgb)
#     weight = torch.norm(rendered_color - 0.5, dim=-1, keepdim=True)  # 遠離中性灰色的像素給較高權重
#     loss_render = (loss_render * weight).mean()

#     # === Debug ===
#     print("Rendered color stats:", rendered_color.min().item(), rendered_color.max().item())
#     print("GT color stats:", gt_rgb.min().item(), gt_rgb.max().item())
#     print("Valid rays:", valid.sum().item())

#     full_img = torch.zeros((H, W, 3), device=rendered_color.device)
#     full_img[v, u] = rendered_color
#     full_img_np = full_img.detach().cpu().numpy().clip(0, 1)
#     plt.imsave("rendered_color_debug.png", full_img_np)

#     return loss_render

# def compute_color_consistency_loss(x, f_x, alpha=10.0, sample_size=1024):
#     N = x.shape[0]
#     idx = torch.randperm(N)[:sample_size]
#     x_sub = x[idx]  # [M,6]
#     f_sub = f_x[idx]  # [M]
#     rgb = x_sub[:, 3:]  # [M,3]

#     diff = rgb.unsqueeze(1) - rgb.unsqueeze(0)  # [M,M,3]
#     color_dist2 = (diff ** 2).sum(-1)  # [M,M]
#     w = torch.exp(-alpha * color_dist2)

#     sdf_diff2 = (f_sub.unsqueeze(1) - f_sub.unsqueeze(0)) ** 2  # [M,M]
#     loss = (w * sdf_diff2).mean()
#     return 100 * loss

# refer to SparseNeuS
def compute_sparse_loss(model, x, num_samples=10000, box_margin=0.1, tau=20.0):
    with torch.no_grad():
        min_bound = x.min(dim=0)[0]
        max_bound = x.max(dim=0)[0]
        margin = (max_bound - min_bound) * box_margin
        min_bound -= margin
        max_bound += margin

        # Uniform sampling in expanded 6D bounding box
        uniform_6d = torch.rand((num_samples, 6), device=x.device) * (max_bound - min_bound) + min_bound
    sdf_pred = model(uniform_6d)
    sparse_loss = torch.exp(-tau * sdf_pred.abs()).mean()
    return sparse_loss

def compute_normal_loss(model, x, normals, batch_size=8192):
    N = x.shape[0]
    total_loss = 0.0
    device = x.device

    for i in range(0, N, batch_size):
        x_batch = x[i:i+batch_size].clone().detach().requires_grad_(True)
        normals_batch = normals[i:i+batch_size]

        f_pred = model(x_batch)
        grads = torch.autograd.grad(
            outputs=f_pred,
            inputs=x_batch,
            grad_outputs=torch.ones_like(f_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, :3]  # normal loss只對xyz，這裡選前3維

        # 只計算靠近表面的normal loss
        mask = (f_pred.abs() < 0.05)
        if mask.any():
            grads = grads[mask]
            normals_batch = normals_batch[mask]
            grads = F.normalize(grads, dim=1)
            normals_batch = F.normalize(normals_batch, dim=1)
            cos_sim = F.cosine_similarity(grads, normals_batch, dim=1)
            loss = ((1 - cos_sim) ** 2).mean()
            total_loss += loss * len(grads)
    return total_loss / N

# total loss calculation
def compute_loss(
        model, 
        x, 
        x_noisy_full, 
        epsilon, 
        normals, 
        H, 
        W, 
        K, 
        cam_pose, 
        gt_image, 
        is_a100, 
        weight_sdf,
        weight_zero,
        eik_init,
        weight_normal,
        weight_sparse
    ):
    # 1. SDF Loss: x_noisy_full 是 6D 點, epsilon 是 6D noise
    if weight_sdf > 0.0:
        # 假設normals為[N,3]，如果你要6D signed distance可直接用
        # loss_sdf = F.mse_loss(model(x_noisy_full), epsilon.norm(dim=1))
        # 或如果你有表面法向（與原NeuS一致）：
        normals_t = torch.tensor(normals, dtype=torch.float32, device=x.device)
        signed_dist = torch.sum(epsilon[:, :3] * normals_t, dim=1)
        loss_sdf = F.mse_loss(model(x_noisy_full), signed_dist)
    else:
        loss_sdf = torch.tensor(0.0, device=x.device)

    # 2. Zero loss: ||f(x, c)|| at GT表面
    if weight_zero > 0.0:
        loss_zero = model(x).abs().mean()
    else:
        loss_zero = torch.tensor(0.0, device=x.device)

    # 3. Eikonal Loss: on 6D (or你只針對xyz可：[:, :3])
    if eik_init > 0.0:
        x_noisy_full = x_noisy_full.requires_grad_(True)
        f_pred = model(x_noisy_full)
        grads = torch.autograd.grad(
            outputs=f_pred,
            inputs=x_noisy_full,
            grad_outputs=torch.ones_like(f_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # 你可以選6D norm，或只對xyz
        grad_norm = grads.norm(dim=-1)
        sdf_abs = torch.abs(f_pred.detach().squeeze(-1))
        mask = sdf_abs < 0.2
        if mask.any():
            loss_eikonal = ((grad_norm[mask] - 1) ** 2).mean()
        else:
            loss_eikonal = torch.tensor(0.0, device=f_pred.device)
    else:
        loss_eikonal = torch.tensor(0.0, device=x.device)

    # 4. Normal Loss
    if weight_normal > 0.0:
        if not isinstance(normals, torch.Tensor):
            normals = torch.tensor(normals, dtype=torch.float32, device=x.device)
        if is_a100:
            loss_normal = compute_normal_loss(model, x, normals, batch_size=8192)
        else:
            loss_normal = compute_normal_loss(model, x, normals, batch_size=2048)
        if not isinstance(loss_normal, torch.Tensor):
            loss_normal = torch.tensor(loss_normal, device=x.device)
    else:
        loss_normal = torch.tensor(0.0, device=x.device)

    # 5. Sparse Loss
    if weight_sparse > 0.0:
        if is_a100:
            loss_sparse = compute_sparse_loss(model, x, num_samples=100000)
        else:
            loss_sparse = compute_sparse_loss(model, x, num_samples=3000)
    else:
        loss_sparse = torch.tensor(0.0, device=x.device)

    return loss_sdf, loss_zero, loss_eikonal, loss_normal, loss_sparse
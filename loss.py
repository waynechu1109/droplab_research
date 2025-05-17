import torch
import torch.nn.functional as F

# def compute_color_smoothness_loss(x, f_x_grad, radius=0.05):
#     # x: [N,6] = [xyz, rgb], f_x_grad: [N,3]
#     pos = x[:, :3]
#     rgb = x[:, 3:]
    
#     # 隨機 sample 周圍點差值（或預先建立鄰接結構）
#     # 這裡以 torch.diff 模擬 RGB 局部 gradient
#     rgb_var = torch.var(rgb, dim=1, unbiased=False)  # [N]
    
#     gamma = rgb_var / (rgb_var.max() + 1e-6)  # normalize 到 [0, 1]
    
#     grad_norm2 = f_x_grad.norm(dim=1) ** 2  # [N]
    
#     return (gamma * grad_norm2).mean()

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

        f_pred = model(x_batch)

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
        # total_loss += loss.item() * len(x_batch)
        total_loss += loss * len(x_batch)

    # return torch.tensor(total_loss / N, device=device)
    return total_loss / N

def compute_edge_loss(x: torch.Tensor,
                      f_x: torch.Tensor,
                      k: int = 8,
                      alpha: float = 0.1,
                      chunk_size: int = 2048) -> torch.Tensor:
    """
    颜色保边正则：
      L_edge = mean_{i,j in N(i)} [ exp(-α||c_i - c_j||^2) * (f_x[i] - f_x[j])^2 ]
    Args:
      x:    [N,6] = [xyz, rgb], 数据点和对应颜色
      f_x:  [N]   = 对应位置的 SDF 预测
      k:     int  = 每点最近邻数量
      alpha: float= 颜色相似度衰减系数
    Returns:
      loss_edge: 标量 Tensor
    """
    # 1) extract position & color
    pts = x[:, :3]   # [N,3]
    rgb = x[:, 3:].float() / 255.0 
    N = pts.shape[0]

    # 2) 构建 k-NN 邻域（排除自身）
    nn_idx_chunks = []
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        # [chunk_size, 3] vs [N, 3]
        dist_chunk = torch.cdist(pts[i:end], pts)     # [b, N]
        _, idx = dist_chunk.topk(k+1, dim=1, largest=False)
        nn_idx_chunks.append(idx[:, 1:])              # 去掉自己
    nn_idx = torch.cat(nn_idx_chunks, dim=0)          # [N, k]

    # 3) 展平成对索引
    N = x.shape[0]
    i_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
    j_idx = nn_idx.reshape(-1)

    # 4) 取出对应颜色和 SDF
    c_i = rgb[i_idx]    # [N*k,3]
    c_j = rgb[j_idx]    # [N*k,3]
    f_i = f_x[i_idx]    # [N*k]
    f_j = f_x[j_idx]    # [N*k]

    # 5) 计算权重 & 差值
    color_diff2 = torch.sum((c_i - c_j)**2, dim=-1)           # [N*k]
    # w = torch.exp(-alpha * color_diff2)                       # [N*k]
    w = torch.clamp(1.0 - color_diff2, min=0.0)
    sdf_diff2 = (f_i - f_j)**2                                # [N*k]

    # 6) 加权平均
    loss_edge = (w * sdf_diff2).mean()
    return 10000*loss_edge

# total loss calculation
def compute_loss(model, x, x_noisy_full, epsilon, normals):
    x.requires_grad_(True)
    f_x = model(x)              # f(x, c) 應接收完整 6 維輸入
    f_x_noisy = model(x_noisy_full)

    # Part 1: MSE between predicted SDF at xⁿ and actual |ε|
    true_dist = epsilon.norm(dim=1)
    loss_sdf = F.mse_loss(f_x_noisy, true_dist)

    # Part 2: ||f(x, c)||
    loss_zero = f_x.abs().mean()

    # Part 3: Eikonal: ||∇f(x^n, c) - 1||
    x_noisy_full.requires_grad_(True) # enable partial differentiation
    # x.requires_grad_()
    f_pred = model(x_noisy_full)
    # f_pred = model(x)
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
    # loss_eikonal = ((grad_norm - 1) ** 2).mean()

    # narrow band mask: 只在靠近表面的位置計算 eikonal loss
    sdf_abs = torch.abs(f_pred.detach().squeeze(-1))  # detach 避免回傳梯度
    mask = sdf_abs < 0.05  # threshold 可調整：0.05～0.2 之間都可嘗試
    if mask.any():
        loss_eikonal = ((grad_norm[mask] - 1) ** 2).mean()
    else:
        loss_eikonal = torch.tensor(0.0, device=f_pred.device)

    # Part 4: Edge loss
    # loss_edge = compute_edge_loss(x, f_x, k=8, alpha=10.0)

    # Part 5: Normal loss
    normals = torch.tensor(normals, dtype=torch.float32, device=x.device)
    loss_normal = compute_normal_loss(model, x, normals, batch_size=10000)

    # Part 6: Color Consistency loss
    loss_consistency = compute_color_consistency_loss(x, f_x, alpha=20.0, sample_size=2048)

    # Part 7: Color Smoothness loss

    
    # return loss_sdf, loss_zero, loss_eikonal, loss_edge, loss_normal
    return loss_sdf, loss_zero, loss_eikonal, loss_normal, loss_consistency
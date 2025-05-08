import torch
import torch.nn.functional as F

# total loss calculation
def compute_loss(model, x, x_noisy_full, epsilon):
    f_x = model(x)              # f(x, c) 應接收完整 6 維輸入
    f_x_noisy = model(x_noisy_full)

    # Part 1: MSE between predicted SDF at xⁿ and actual |ε|
    true_dist = epsilon.norm(dim=1)
    loss_sdf = F.mse_loss(f_x_noisy, true_dist)

    # Part 2: ||f(x, c)||
    loss_zero = f_x.abs().mean()

    # Part 3: Eikonal: ||∇f(x^n, c) - 1||
    # x_noisy_full.requires_grad_() # enable partial differentiation
    x.requires_grad_()
    # f_pred = model(x_noisy_full)
    f_pred = model(x)
    grads = torch.autograd.grad(
        outputs=f_pred,       # f(xⁿ, c)
        inputs=x,  # input with 6 dimensions: [x, y, z, r, g, b]
        grad_outputs=torch.ones_like(f_pred),
        create_graph=False,
        retain_graph=True,
        only_inputs=True
    )[0][:, :3]  # take the gradieant w.r.t. (x, y, z)

    grad_norm = torch.norm(grads, dim=-1)
    loss_eikonal = ((grad_norm - 1) ** 2).mean()

    # # narrow band mask: 只在靠近表面的位置計算 loss
    # sdf_abs = torch.abs(f_pred.detach().squeeze(-1))  # detach 避免回傳梯度
    # mask = sdf_abs < 0.2  # threshold 可調整：0.05～0.2 之間都可嘗試

    # if mask.any():
    #     loss_eikonal = ((grad_norm[mask] - 1) ** 2).mean()
    # else:
    #     loss_eikonal = torch.tensor(0.0, device=f_pred.device)

    return loss_sdf, loss_zero, loss_eikonal
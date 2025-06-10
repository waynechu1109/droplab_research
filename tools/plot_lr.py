import torch
import matplotlib.pyplot as plt

# === 根據你的 schedule.json 設定修改這些值 ===
coarse_epochs = 1000
fine_epochs = 1000
color_epochs = 500
total_epochs = coarse_epochs + fine_epochs + color_epochs

initial_lr = 0.005
eta_min = 1e-5

# 建立假 optimizer
model = torch.nn.Linear(1, 1)  # dummy model
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)

# Scheduler 1：coarse + fine 階段
scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=coarse_epochs + fine_epochs,
    eta_min=eta_min
)

# Scheduler 2：color 階段
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=color_epochs,
    eta_min=eta_min
)

# 模擬整個 epoch 的 learning rate
lr_values = []
for epoch in range(total_epochs):
    current_lr = optimizer.param_groups[0]['lr']
    lr_values.append(current_lr)

    # 切換 scheduler 條件
    if epoch < coarse_epochs + fine_epochs:
        scheduler_1.step()
    else:
        if epoch == coarse_epochs + fine_epochs:
            for group in optimizer.param_groups:
                group['lr'] = initial_lr  # reset lr
        scheduler_2.step()

# 畫圖但不顯示
plt.figure(figsize=(10, 5))
plt.plot(range(total_epochs), lr_values, label='Learning Rate')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule (CosineAnnealing)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lr_schedule.png")  # 儲存圖像
# plt.show() 不呼叫 => 不會顯示

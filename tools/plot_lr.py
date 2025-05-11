import torch
import matplotlib.pyplot as plt

# Setup a dummy parameter to attach the optimizer
param = torch.nn.Parameter(torch.zeros(1))

# Initialize AdamW optimizer and CosineAnnealingLR scheduler
optimizer = torch.optim.AdamW([param], lr=0.005, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2500, eta_min=1e-5)

# Record learning rates for each epoch
lrs = []
for epoch in range(2500):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Plot the LR schedule
plt.figure()
plt.plot(range(1, 2501), lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing LR Schedule (T_max=2500)')
plt.tight_layout()
# print('show')
# plt.show()
plt.savefig('lr.jpg')
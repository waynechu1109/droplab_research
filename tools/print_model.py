import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchinfo import summary
from model import SDFNet
import torch

model = SDFNet(use_viewdirs=True)
model.eval()

summary(model, 
        input_data=[torch.randn(1, 3), torch.randn(1, 3)],  # [xyz, view_dirs]
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=3)
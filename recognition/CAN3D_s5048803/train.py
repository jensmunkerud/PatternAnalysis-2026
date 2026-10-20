"""
train.py: Containing the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be imported from
dataset.py. Make sure to plot the losses and metrics during training.
"""

import torch
from modules import CAN
import dataset

# 3D: (batch, channels, depth, height, width)
model3d = CAN(in_ch=1, num_classes=6, dim=3)
x3d = torch.randn(1, 1, 32, 64, 64)
with torch.no_grad():
	out3d = model3d(x3d)
print("3D:", x3d.shape, "->", out3d.shape)   # expect (1, 6, 32, 64, 64)

# 2D: (batch, channels, height, width)
model2d = CAN(in_ch=1, num_classes=6, dim=2)
x2d = torch.randn(4, 1, 256, 128)
with torch.no_grad():
	out2d = model2d(x2d)
print("2D:", x2d.shape, "->", out2d.shape)   # expect (4, 6, 256, 128)

# Parameter counts
for name, m in [("2D", model2d), ("3D", model3d)]:
	print(name, sum(p.numel() for p in m.parameters()), "parameters")
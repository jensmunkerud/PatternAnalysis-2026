import torch.nn as nn

class DilatedBlock(nn.Module):
	def __init__(self, in_ch, out_ch, dilation, dim=3):
		super().__init__()
		Conv = nn.Conv3d if dim == 3 else nn.Conv2d
		Norm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
		self.conv = Conv(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
		self.norm = Norm(out_ch)
		self.act = nn.LeakyReLU(0.01, inplace=True)

	def forward(self, x):
		return self.act(self.norm(self.conv(x)))

class CAN(nn.Module):
	"""dim=2 gives the baseline, dim=3 gives CAN3D."""
	def __init__(self, in_ch=1, num_classes=6, width=24,
				 dilations=((1,1,1),(2,2,2),(4,4,2),(8,8,2),(1,1,1)), dim=3):
		super().__init__()
		layers, ch = [], in_ch
		for d in dilations:
			d = d if dim == 3 else d[0]
			layers.append(DilatedBlock(ch, width, d, dim))
			ch = width
		self.body = nn.Sequential(*layers)
		Conv = nn.Conv3d if dim == 3 else nn.Conv2d
		self.head = Conv(width, num_classes, kernel_size=1)

	def forward(self, x):
		return self.head(self.body(x))
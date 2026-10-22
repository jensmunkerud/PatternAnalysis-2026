import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================
# CAN network parameters
# ============================================================

IN_CH = 1
NUM_CLASSES = 6
NETWORK_WIDTH = 24

DILATIONS = [
	(1, 1, 1),
	(2, 2, 2),
	(2, 4, 4),
	(2, 8, 8),
	(1, 1, 1)
]

# ============================================================
# Drawing helpers
# ============================================================

def draw_block(ax, x, y, width, height, title, details):
	box = FancyBboxPatch(
		(x, y),
		width,
		height,
		boxstyle="round,pad=0.02,rounding_size=0.08",
		linewidth=2,
		facecolor="white",
		edgecolor="black"
	)

	ax.add_patch(box)

	ax.text(
		x + width / 2,
		y + height * 0.72,
		title,
		ha="center",
		va="center",
		fontsize=12,
		fontweight="bold"
	)

	ax.text(
		x + width / 2,
		y + height * 0.38,
		details,
		ha="center",
		va="center",
		fontsize=9
	)


def draw_arrow(ax, x1, y1, x2, y2):
	ax.add_patch(
		FancyArrowPatch(
			(x1, y1),
			(x2, y2),
			arrowstyle="->",
			mutation_scale=15,
			linewidth=2
		)
	)


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(18, 6))

ax.set_xlim(0, 18)
ax.set_ylim(0, 6)
ax.axis("off")

# ============================================================
# Input
# ============================================================

draw_block(
	ax,
	0.3, 2.2,
	1.5, 1.6,
	"INPUT",
	"1 channel\nD × H × W"
)

draw_arrow(ax, 1.8, 3.0, 2.2, 3.0)

# ============================================================
# Dilated blocks
# ============================================================

x = 2.2

block_width = 2.5
block_height = 2.4
gap = 0.45

in_ch = IN_CH

for i, dilation in enumerate(DILATIONS):

	title = f"DilatedBlock {i + 1}"

	details = (
		f"Conv3D\n"
		f"3 × 3 × 3\n"
		f"dilation = {dilation}\n"
		f"padding = {dilation}\n\n"
		f"InstanceNorm3D\n"
		f"LeakyReLU"
	)

	draw_block(
		ax,
		x,
		1.8,
		block_width,
		block_height,
		title,
		details
	)

	if i > 0:
		draw_arrow(
			ax,
			x - gap,
			3.0,
			x,
			3.0
		)

	in_ch = NETWORK_WIDTH
	x += block_width + gap

# ============================================================
# Head
# ============================================================

draw_arrow(
	ax,
	x - gap,
	3.0,
	x,
	3.0
)

draw_block(
	ax,
	x,
	2.0,
	2.0,
	2.0,
	"HEAD",
	"1 × 1 × 1 Conv3D\n\n24 → 6 channels"
)

x += 2.0

draw_arrow(
	ax,
	x - 2.0,
	3.0,
	x,
	3.0
)

# ============================================================
# Output
# ============================================================

draw_block(
	ax,
	x,
	2.2,
	1.5,
	1.6,
	"OUTPUT",
	"6 classes\nD × H × W"
)

# ============================================================
# Title
# ============================================================

ax.text(
	9,
	5.4,
	"CAN3D Architecture",
	ha="center",
	va="center",
	fontsize=20,
	fontweight="bold"
)

ax.text(
	9,
	0.6,
	"Each DilatedBlock: Conv3D → InstanceNorm3D → LeakyReLU",
	ha="center",
	va="center",
	fontsize=11
)

plt.tight_layout()
# plt.show()
plt.savefig("modelArchitecture")
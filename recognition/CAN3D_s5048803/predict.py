"""
predict.py: Showing example usage of your trained model. It should load your saved model
weights, run inference on test cases, print out any results, and provide visualisations where applicable
(e.g., prediction overlays, generated samples, heatmaps). NumPy is allowed in this script for
visualization or loading data.

Usage:
	python predict.py --model can3d --weights outputs/can3d_best.pt
	python predict.py --model can2d --weights outputs/can2d_best.pt
"""

import argparse, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules import CAN
from dataset import (get_pairs, split_pairs, HipMRI3D, get_slices, HipMRI2D,
					 NUM_CLASSES, CLASS_NAMES, RANGPUR_MODE)

def get_device():
	return ("cuda" if torch.cuda.is_available()
			else "mps" if torch.backends.mps.is_available() else "cpu")

def load_model(weights, dim, device):
	"""Rebuild the architecture, then load the trained weights into it."""
	model = CAN(in_ch=1, num_classes=NUM_CLASSES, dim=dim).to(device)
	model.load_state_dict(torch.load(weights, map_location=device))
	return model.eval()

def load_test_set(dim, root, scale):
	if dim == 3:
		return HipMRI3D(split_pairs(root, get_pairs(root))["test"], scale)
	return HipMRI2D(get_slices(root, "test"), scale)


def confusion(pred, target):
	"""NUM_CLASSES x NUM_CLASSES counts. Rows = ground truth class, columns = predicted class."""
	idx = target.flatten() * NUM_CLASSES + pred.flatten()
	return torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

def metrics(conf):
	"""Per-class Dice, IoU, over-segmentation % and under-segmentation % from a confusion matrix.
	over  = predicted voxels outside the ground truth, as % of the ground-truth size (can exceed 100)
	under = ground-truth voxels the model missed, as % of the ground-truth size"""
	tp = conf.diag().float()
	fp = conf.sum(0).float() - tp
	fn = conf.sum(1).float() - tp
	gt = (tp + fn).clamp(min=1)
	return dict(dice=2 * tp / (2 * tp + fp + fn).clamp(min=1),
				iou=tp / (tp + fp + fn).clamp(min=1),
				over=100 * fp / gt,
				under=100 * fn / gt)

@torch.no_grad()
def evaluate(model, ds, dim, device):
	"""Pooled confusion matrix over a whole test set (batches of slices for 2D, one volume for 3D)."""
	conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
	for x, y in DataLoader(ds, batch_size=32 if dim == 2 else 1):
		conf += confusion(model(x.to(device)).argmax(1), y.to(device))
	return conf.cpu()

def print_table(name, m):
	print(f"{name}, test set:")
	for c in range(NUM_CLASSES):
		print(f"  {CLASS_NAMES[c]:>8}: Dice {m['dice'][c]:.3f}  IoU {m['iou'][c]:.3f}  "
			  f"over {m['over'][c]:.1f}%  under {m['under'][c]:.1f}%")
	print(f"  mean foreground: Dice {m['dice'][1:].mean():.3f}  IoU {m['iou'][1:].mean():.3f}")


def pick_cases(ds, cls, n):
	"""Indices of n items containing `cls`, spread evenly across the test set."""
	have = [i for i in range(len(ds)) if (ds[i][1] == cls).any()]
	assert have, f"No test item contains class {cls}"
	return have[::max(1, len(have) // n)][:n]

@torch.no_grad()
def run_cases(model, ds, dim, label, cls, n, device):
	"""Predict n example items. Case stats cover the whole item: the full volume for 3D, the slice for 2D."""
	cases = []
	for i in pick_cases(ds, cls, n):
		x, y = ds[i]
		pred = model(x[None].to(device)).argmax(1)[0].cpu()
		m = metrics(confusion(pred, y))
		if dim == 3:                                   # show the slice with most of the class
			s = (y == cls).sum((1, 2)).argmax()
			img, gt, pr = x[0, s], y[s], pred[s]
		else:
			img, gt, pr = x[0], y, pred
		cases.append(dict(label=label, index=i, img=img, gt=gt, pred=pr, iou=m["iou"][cls].item(),
						  over=m["over"][cls].item(), under=m["under"][cls].item()))
	return cases

def plot_cases(cases, path):
	"""One row per case: image | ground truth | prediction | difference (red = wrong)."""
	fig, ax = plt.subplots(len(cases), 4, figsize=(12, 3 * len(cases)), squeeze=False)
	for r, c in enumerate(cases):
		ax[r, 0].imshow(c["img"], cmap="gray")
		ax[r, 1].imshow(c["gt"], cmap="tab10", vmin=0, vmax=9)
		ax[r, 2].imshow(c["pred"], cmap="tab10", vmin=0, vmax=9)
		ax[r, 3].imshow(c["img"], cmap="gray")
		ax[r, 3].imshow((c["pred"] != c["gt"]).float(), cmap="Reds", alpha=0.6)
		titles = [f'{c["label"]} image', "Ground truth", "Prediction",
				  f'IoU {c["iou"]:.2f} | over {c["over"]:.0f}% | under {c["under"]:.0f}%']
		for a, t in zip(ax[r], titles):
			a.set_title(t, fontsize=9)
			a.axis("off")
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)


def plot_dice_comparison(m2d, m3d, path):
	"""Grouped bar plot: per-class Dice of CAN2D next to CAN3D."""
	x = np.arange(NUM_CLASSES)
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.bar(x - 0.2, m2d["dice"].numpy(), 0.4, label="CAN2D")
	ax.bar(x + 0.2, m3d["dice"].numpy(), 0.4, label="CAN3D")
	ax.set_xticks(x)
	ax.set_xticklabels(CLASS_NAMES)
	ax.set_ylabel("Dice")
	ax.set_ylim(0, 1)
	ax.set_title("Per-class Dice on the test set")
	ax.legend()
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--weights2d", default="./outputs/can2d_best.pt", help="path to the CAN2D .pt file")
	ap.add_argument("--weights3d", default="./outputs/can3d_best.pt", help="path to the CAN3D .pt file")
	ap.add_argument("--scale2d", type=float, default=0.5, help="must match CAN2D training")
	ap.add_argument("--scale3d", type=float, default=0.5, help="must match CAN3D training")
	ap.add_argument("--cls", type=int, default=5, help="class shown in the example cases (assumed prostate)")
	ap.add_argument("--n", type=int, default=2, help="example cases per model")
	ap.add_argument("--out", default="outputs")
	ap.add_argument("--data_root", default="/home/groups/comp3710/HipMRI_Study_open" if RANGPUR_MODE else ".")
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)
	device = get_device()

	# evaluate both models on their own test sets
	runs = {}
	for dim, weights, scale in ((3, args.weights3d, args.scale3d), (2, args.weights2d, args.scale2d)):
		model = load_model(weights, dim, device)
		ds = load_test_set(dim, args.data_root, scale)
		m = metrics(evaluate(model, ds, dim, device))
		runs[dim] = dict(label=f"CAN{dim}D", model=model, ds=ds, metrics=m)
		print_table(f"CAN{dim}D", m)

	# example cases: CAN3D on the top rows, CAN2D below
	cases = []
	for dim in (3, 2):
		r = runs[dim]
		cases += run_cases(r["model"], r["ds"], dim, r["label"], args.cls, args.n, device)
	for c in cases:
		print(f'{c["label"]} item {c["index"]}, {CLASS_NAMES[args.cls]}: '
			  f'IoU {c["iou"]:.3f}  over {c["over"]:.1f}%  under {c["under"]:.1f}%')
	plot_cases(cases, os.path.join(args.out, "cases.png"))

	# per-class Dice, 2D vs 3D
	plot_dice_comparison(runs[2]["metrics"], runs[3]["metrics"], os.path.join(args.out, "dice_comparison.png"))
	print("saved cases.png and dice_comparison.png to", args.out)

if __name__ == "__main__":
	main()
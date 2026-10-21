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
import torch
import matplotlib.pyplot as plt
from modules import CAN
from dataset import (get_pairs, split_pairs, HipMRI3D, get_slices, HipMRI2D,
					 NUM_CLASSES, RANGPUR_MODE)

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--model", choices=["can2d", "can3d"], required=True)
	ap.add_argument("--weights", required=True, help="path to the .pt file")
	ap.add_argument("--scale", type=float, default=None, help="must match training (default 0.5 for 3D, 1.0 for 2D)")
	ap.add_argument("--cls", type=int, default=5, help="class to pick example slices for (assumed prostate)")
	ap.add_argument("--n", type=int, default=4, help="number of example rows")
	ap.add_argument("--out", default="outputs")
	ap.add_argument("--data_root", default="/home/groups/comp3710/HipMRI_Study_open" if RANGPUR_MODE else ".")
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)

	dim = 3 if args.model == "can3d" else 2
	scale = args.scale if args.scale is not None else (0.5 if dim == 3 else 1.0)
	device = ("cuda" if torch.cuda.is_available()
			  else "mps" if torch.backends.mps.is_available() else "cpu")

	# test split only
	if dim == 3:
		ds = HipMRI3D(split_pairs(args.data_root, get_pairs(args.data_root))["test"], scale)
	else:
		ds = HipMRI2D(get_slices(args.data_root, "test"), scale)

	# rebuild the architecture, then load the trained weights into it
	model = CAN(in_ch=1, num_classes=NUM_CLASSES, dim=dim).to(device)
	model.load_state_dict(torch.load(args.weights, map_location=device))
	model.eval()

	inter, pred_n, true_n = (torch.zeros(NUM_CLASSES) for _ in range(3))
	examples = []
	with torch.no_grad():
		for i in range(len(ds)):
			x, y = ds[i]
			pred = model(x[None].to(device)).argmax(1)[0].cpu()
			for c in range(NUM_CLASSES):                  # pooled counts for Dice
				p, t = pred == c, y == c
				inter[c] += (p & t).sum(); pred_n[c] += p.sum(); true_n[c] += t.sum()
			if len(examples) < args.n and (y == args.cls).any():
				if dim == 3:                              # pick the slice with most of the class
					s = (y == args.cls).sum((1, 2)).argmax()
					examples.append((x[0, s], y[s], pred[s]))
				else:
					examples.append((x[0], y, pred))

	dice = (2 * inter + 1e-6) / (pred_n + true_n + 1e-6)
	print(f"{args.model} on {len(ds)} test items")
	for c, d in enumerate(dice):
		print(f"  class {c}: Dice {d:.3f}")
	print(f"  mean foreground Dice: {dice[1:].mean():.3f}")

	# figure: image | ground truth | prediction | difference
	fig, ax = plt.subplots(len(examples), 4, figsize=(12, 3 * len(examples)), squeeze=False)
	for r, (img, gt, pr) in enumerate(examples):
		ax[r, 0].imshow(img, cmap="gray")
		ax[r, 1].imshow(gt, cmap="tab10", vmin=0, vmax=9)
		ax[r, 2].imshow(pr, cmap="tab10", vmin=0, vmax=9)
		ax[r, 3].imshow(img, cmap="gray")
		ax[r, 3].imshow((pr != gt).float(), cmap="Reds", alpha=0.6)   # red = wrong voxels
		for a, t in zip(ax[r], ["Image", "Ground truth", "Prediction", "Difference"]):
			a.set_title(t); a.axis("off")
	fig.tight_layout()
	path = os.path.join(args.out, f"{args.model}_predictions.png")
	fig.savefig(path)
	print("saved", path)

if __name__ == "__main__":
	main()
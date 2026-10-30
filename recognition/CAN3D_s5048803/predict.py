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
import time
import torch
import resource, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from modules import CAN
from dataset import (get_pairs, split_pairs, HipMRI3D, get_slices, HipMRI2D, key, slice_key,
					 NUM_CLASSES, CLASS_NAMES, FG_NAMES, RANGPUR_MODE)

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

def metrics_from_conf(conf):
	"""Calculate per-class segmentation metrics from confusion matrix"""
	tp = conf.diag().float()
	fp = conf.sum(0).float() - tp
	fn = conf.sum(1).float() - tp
	gt = (tp + fn).clamp(min=1)
	result = dict(dice=2 * tp / (2 * tp + fp + fn).clamp(min=1),
				  iou=tp / (tp + fp + fn).clamp(min=1),
				  over=100 * fp / gt,
				  under=100 * fn / gt)
	return {k: v[1:] for k, v in result.items()}

def metrics(pred, target):
	"""Get metrics from prediction and target"""
	return metrics_from_conf(confusion(pred, target))

@torch.no_grad()
def evaluate(model, dataset, dim, device):
	"""Pooled confusion matrix over a whole test set (batches of slices for 2D, one volume for 3D)."""
	conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

	for x, y in DataLoader(dataset, batch_size=32 if dim == 2 else 1):
		pred = model(x.to(device)).argmax(1)
		conf += confusion(pred, y.to(device))
	return conf.cpu()

def print_table(name, m):
	print(f"{name}, test set:")
	for i, cname in enumerate(FG_NAMES):
		print(f"  {cname:>8}: Dice {m['dice'][i]:.3f}  IoU {m['iou'][i]:.3f}  "
			  f"over {m['over'][i]:.1f}%  under {m['under'][i]:.1f}%")
	print(f"  mean foreground: Dice {m['dice'][1:].mean():.3f}  IoU {m['iou'][1:].mean():.3f}")


def pick_cases(ds, cls, n):
	"""Indices of n items containing `cls`, spread evenly across the test set."""
	have = [i for i in range(len(ds)) if (ds[i][1] == cls).any()]
	assert have, f"No test item contains class {cls}"
	return have[::max(1, len(have) // n)][:n]

@torch.no_grad()
def plot_item(model, ds, index, device, path):
	x, y = ds[index]

	pred = model(x[None].to(device)).argmax(1)[0].cpu()

	fig, ax = plt.subplots(1, 3, figsize=(12, 4))

	ax[0].imshow(x[0], cmap="gray")
	ax[0].set_title(f"CAN2D item {index} - MRI")

	ax[1].imshow(y, cmap="Reds")
	ax[1].set_title("Ground truth")

	ax[2].imshow(pred, cmap="Reds")
	ax[2].set_title("Prediction")

	for a in ax:
		a.axis("off")

	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)


@torch.no_grad()
def run_cases(model, ds, dim, label, cls, n, device):
	"""Predict n example items. Case stats cover the whole item: the full volume for 3D, the slice for 2D."""
	cases = []
	for i in pick_cases(ds, cls, n):
		x, y = ds[i]
		pred = model(x[None].to(device)).argmax(1)[0].cpu()
		m = metrics(confusion(pred, y))
		if dim == 3: # show the slice with most of the class
			s = (y == cls).sum((1, 2)).argmax()
			img, gt, pr = x[0, s], y[s], pred[s]
		else:
			img, gt, pr = x[0], y, pred
		cases.append(dict(label=label, index=i, img=img, gt=gt, pred=pr, iou=m["iou"][cls-1].item(),
						  over=m["over"][cls-1].item(), under=m["under"][cls-1].item()))
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
	x = np.arange(len(FG_NAMES))
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.bar(x - 0.2, m2d["dice"].numpy(), 0.4, label="CAN2D")
	ax.bar(x + 0.2, m3d["dice"].numpy(), 0.4, label="CAN3D")
	ax.set_xticks(x)
	ax.set_xticklabels(FG_NAMES)
	ax.set_ylabel("Dice")
	ax.set_ylim(0.5, 1)
	ax.set_title("Per-class Dice on the test set")
	ax.legend()
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)


def mem_mb(device):
	"""Memory in MB: CUDA = peak allocated; MPS = memory held by the Metal driver; CPU = peak process RSS."""
	if device == "cuda":
		return torch.cuda.max_memory_allocated() / 1024**2
	if device == "mps":
		return torch.mps.driver_allocated_memory() / 1024**2
	rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # bytes on macOS, KB on Linux
	return rss / 1024**2 if sys.platform == "darwin" else rss / 1024

@torch.no_grad()
def time_mem_inference(model, x, device, warmup=3, runs=10):
	"""Mean/std seconds for one forward pass of x (shaped [B, C, ...]) and peak GPU memory in MB (NaN off CUDA)."""
	x = x.to(device)
	sync = torch.cuda.synchronize if device == "cuda" else (torch.mps.synchronize if device == "mps" else (lambda: None))
	if device == "mps":
		torch.mps.empty_cache()
	for _ in range(warmup): # warm-up
		model(x)
	sync()
	if device == "cuda":
		torch.cuda.reset_peak_memory_stats()   # count only the timed runs, starting from weights + input
	times = []
	peak_mem = 0.0
	for _ in range(runs):
		sync()
		t0 = time.perf_counter()
		model(x)
		sync()
		times.append(time.perf_counter() - t0)
		peak_mem = max(peak_mem, mem_mb(device))
	t = torch.tensor(times)
	return t.mean().item(), t.std().item(), peak_mem

def num_parameters(model, dim):
	total_params = sum(p.numel() for p in model.parameters())
	print(f"Total parameters {dim}D: {total_params:,}")


@torch.no_grad()
def slice_metrics(pred, y):
	results = {
		"dice": [],
		"iou": [],
		"over": [],
		"under": []
	}

	for layer in range(len(y)):
		metric_results = metrics(pred[layer], y[layer])

		for name in results:
			results[name].append(metric_results[name])

	return {
		name: torch.stack(values)
		for name, values in results.items()
	}

def plot_slice_metrics(results2d, results3d, cls, x):
	metrics_to_plot = ["dice", "iou", "over", "under"]
	titles = ["Dice", "IoU", "Over-contouring", "Under-contouring"]

	fig, ax = plt.subplots(2, 2, figsize=(12, 8))
	ax = ax.flatten()

	for a, metric, title in zip(ax, metrics_to_plot, titles):
		values2d = results2d[metric][:, cls - 1].numpy()
		values3d = results3d[metric][:, cls - 1].numpy()
		if (metric == "dice"):
			a.axhline(y=0.7, color='r')

		a.plot(x, values2d, color='blue', label="CAN2D")
		a.plot(x, values3d, color='green', label="CAN3D")
		a.set_title(f"Per-layer {title}")
		a.set_xlabel("Slice")
		a.set_ylabel("%" if metric in ("over", "under") else metric)
		a.grid()
		a.legend()

		if metric in ("dice", "iou"):
			a.set_ylim(0, 1)

	for a in ax:
		a.xaxis.set_major_locator(MaxNLocator(integer=True))

	fig.tight_layout()
	fig.savefig("./results/can_depth_comparison.png", dpi=200)
	plt.close(fig)

# Add a a mask showing where the ground truth class is in the x axis
@torch.no_grad()
def compare_slice_metrics(runs, root, vol, cls, device, scale2d):
	ds3 = runs[3]["ds"]

	matched = []

	# Match the 2d slices with the 3d volumes
	for i, pair in enumerate(ds3.pairs):
		slices = matching_slices(root, pair)

		if slices:
			matched.append((i, slices))

	i3, slices = matched[vol]

	x3, y3 = ds3[i3]

	pred3 = runs[3]["model"](
		x3[None].to(device)
	).argmax(1)[0].cpu()

	ds2 = HipMRI2D(slices, scale2d)

	x2 = torch.stack([ds2[i][0] for i in range(len(ds2))])
	y2 = torch.stack([ds2[i][1] for i in range(len(ds2))])

	pred2 = runs[2]["model"](
		x2.to(device)
	).argmax(1).cpu()

	y2 = y2[..., :64] # Drop the padding
	pred2  = pred2[..., :64]
	
	nums = torch.tensor([int(slice_key(p[0])[2]) for p in slices])

	# 2d dataset is cut along axis 2 of the 3d dataset
	def planes(t):
		return t.movedim(2, 0).transpose(1, 2)      # (n_planes, 128, 64), same as a 2D slice

	v = planes(y3)
	score = (y2[:, None] == v[None]).float().mean((2, 3))   # (n2d, n_planes)
	idx = score.argmax(1)


	y3, pred3 = planes(y3)[idx], planes(pred3)[idx]

	results2d = slice_metrics(pred2, y2)
	results3d = slice_metrics(pred3, y3)

	plot_slice_metrics(
		results2d,
		results3d,
		cls,
		nums
	)


def matching_slices(root, pair3d):
	"""The keras test slices from the same patient and week as one 3D volume, ordered by slice number."""
	pid, week = key(pair3d[0])
	sl = [p for p in get_slices(root, "test")
		  if slice_key(p[0])[0] == pid and slice_key(p[0])[1] == week]
	return sorted(sl, key=lambda p: int(slice_key(p[0])[2]))



def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--weights2d", default="./outputs/can2d_best.pt", help="path to the CAN2D .pt file")
	ap.add_argument("--weights3d", default="./outputs/can3d_best.pt", help="path to the CAN3D .pt file")
	ap.add_argument("--scale2d", type=float, default=0.5, help="must match CAN2D training")
	ap.add_argument("--scale3d", type=float, default=0.5, help="must match CAN3D training")
	ap.add_argument("--cls", type=int, default=5, help="Class to plot metrics of (5=prostate)")
	ap.add_argument("--n", type=int, default=2, help="example cases per model")
	ap.add_argument("--out", default="results", help="Folder to put model results")
	ap.add_argument("--data_root", default="/home/groups/comp3710/HipMRI_Study_open" if RANGPUR_MODE else ".")
	ap.add_argument("--vol", type=int, default=1, help="test volume index for the per-slice plot")
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)
	device = get_device()

	# Evaluate both models' Dice, IoU, over -and under contouring for every class (except background)
	runs = {}
	for dim, weights, scale in ((3, args.weights3d, args.scale3d), (2, args.weights2d, args.scale2d)):
		model = load_model(weights, dim, device)
		dataset = load_test_set(dim, args.data_root, scale)

		if dim == 2:
			plot_item(model,dataset,200,device,"results/can2d_item7.png")

		metric_results = metrics_from_conf(evaluate(model, dataset, dim, device))
		runs[dim] = dict(label=f"CAN{dim}D", model=model, ds=dataset, metrics=metric_results)
		print_table(f"CAN{dim}D", metric_results)

	compare_slice_metrics(
	runs,
	args.data_root,
	args.vol,
	args.cls,
	device,
	args.scale2d
	)

	# Visualization of 2D vs 3D segmentation
	# cases = []
	# for dim in (3, 2):
	# 	r = runs[dim]
	# 	cases += run_cases(r["model"], r["ds"], dim, r["label"], args.cls, args.n, device)
	# for c in cases:
	# 	print(f'{c["label"]} item {c["index"]}, {CLASS_NAMES[args.cls]}: '
	# 		  f'IoU {c["iou"]:.3f}  over {c["over"]:.1f}%  under {c["under"]:.1f}%')
	# plot_cases(cases, os.path.join(args.out, "cases.png"))

	# per-class Dice, 2D vs 3D
	# plot_dice_comparison(runs[2]["metrics"], runs[3]["metrics"], os.path.join(args.out, "dice_comparison.png"))

	# Run inference time and memory usage for both models
	# x3d = runs[3]["ds"][0][0][None]    # (1, 1, D, H, W): one test volume
	# x2d = x3d[0].permute(1, 0, 2, 3)   # (D, 1, H, W): the same volume as D slices in one batch

	# for name, model, x in (("CAN3D", runs[3]["model"], x3d), ("CAN2D", runs[2]["model"], x2d)):
	# 	mean, std, mem = time_mem_inference(model, x, device)
	# 	print(f"{name}: {mean*1000:.1f} +/- {std*1000:.1f} ms per volume, peak {mem:.0f} MB")


	# compare_slice_dice(runs, args.data_root, args.vol, args.cls,
				#    os.path.join(args.out, "slice_dice.png"), device, args.scale2d, args.scale3d)
if __name__ == "__main__":
	main()
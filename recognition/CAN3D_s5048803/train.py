"""
train.py: Containing the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be imported from
dataset.py. Make sure to plot the losses and metrics during training.
"""

import argparse, os, time
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules import CAN
from dataset import get_paths, split_by_patient, HipMRI3D, NUM_CLASSES

RANGPUR = "/home/groups/comp3710/HipMRI_Study_open"

def dice_loss(logits, target, eps=1e-6):
	probs = torch.softmax(logits, 1)
	onehot = F.one_hot(target, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
	dims = (0, 2, 3, 4)
	inter = (probs * onehot).sum(dims)
	denom = probs.sum(dims) + onehot.sum(dims)
	return 1 - ((2 * inter + eps) / (denom + eps)).mean()

@torch.no_grad()
def per_class_dice(logits, target, eps=1e-6):
	pred = logits.argmax(1)
	d = []
	for c in range(NUM_CLASSES):
		p, t = pred == c, target == c
		d.append(((2 * (p & t).sum() + eps) / (p.sum() + t.sum() + eps)).item())
	return d

def evaluate(model, loader, device):
	model.eval()
	tot = torch.zeros(NUM_CLASSES)
	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			tot += torch.tensor(per_class_dice(model(x), y))
	return (tot / len(loader)).tolist()

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_root", default=os.environ.get("HIPMRI_ROOT", RANGPUR))
	ap.add_argument("--epochs", type=int, default=50)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--scale", type=float, default=0.5)
	ap.add_argument("--overfit", type=int, default=0, help="train+val on N volumes")
	ap.add_argument("--out", default="outputs")
	args = ap.parse_args()
	os.makedirs(args.out, exist_ok=True)

	device = ("cuda" if torch.cuda.is_available()
			  else "mps" if torch.backends.mps.is_available() else "cpu")
	imgs, lbls = get_paths(args.data_root)
	splits = split_by_patient(imgs, lbls)
	if args.overfit:
		splits["train"] = splits["val"] = splits["train"][:args.overfit]

	train_dl = DataLoader(HipMRI3D(splits["train"], args.scale), batch_size=1, shuffle=True, num_workers=2)
	val_dl = DataLoader(HipMRI3D(splits["val"], args.scale), batch_size=1)

	model = CAN(in_ch=1, num_classes=NUM_CLASSES, dim=3).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=args.lr)

	losses, val_dice, best = [], [], 0
	for ep in range(args.epochs):
		model.train(); run = 0; t0 = time.time()
		for x, y in train_dl:
			x, y = x.to(device), y.to(device)
			logits = model(x)
			loss = F.cross_entropy(logits, y) + dice_loss(logits, y)
			opt.zero_grad(); loss.backward(); opt.step()
			run += loss.item()
		d = evaluate(model, val_dl, device)
		fg = sum(d[1:]) / (NUM_CLASSES - 1)          # mean Dice excluding background
		losses.append(run / len(train_dl)); val_dice.append(fg)
		print(f"ep {ep+1}: loss {losses[-1]:.4f} val fg-dice {fg:.3f} "
			  f"per-class {[round(v, 2) for v in d]} ({time.time()-t0:.0f}s)")
		if fg > best:
			best = fg; torch.save(model.state_dict(), os.path.join(args.out, "can3d_best.pt"))

	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].plot(losses); ax[0].set_title("Train loss")
	ax[1].plot(val_dice); ax[1].set_title("Val mean foreground Dice")
	fig.savefig(os.path.join(args.out, "curves.png"))

if __name__ == "__main__":
	main()
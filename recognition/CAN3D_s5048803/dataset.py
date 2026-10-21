"""
dataset.py: Containing the data loader for loading and preprocessing your data, including leakagefree train/validation/test splitting functions and data augmentations.
"""

import os, glob, re, random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

NUM_CLASSES = 6

# Do export RANGPUR_MODE=1 in the slurm script before calling the python code to set correct filepaths
RANGPUR_MODE = os.environ.get("RANGPUR_MODE", "0") == "1"

# Choosing between the rangpur filelayout or having downloaded the datasets locally
LAYOUTS = {
	"local": dict(d2="HipMRI_2D", mr="HipMRI_3D/semantic_MRs_anon",
				lbl="HipMRI_3D/semantic_labels_anon"),
	"rangpur": dict(d2="keras_slices_data", mr="semantic_MRs",
					lbl="semantic_labels_only"),
}

def layout():
	return LAYOUTS["rangpur" if RANGPUR_MODE else "local"]

def key(path):
	"""'Case_004_Week0_LFOV.nii.gz' or 'B006_Week1_SEMANTIC.nii.gz' -> ('004'/'006', '0'/'1')."""
	m = re.search(r"(?:Case_|B)(\d+)_Week(\d+)", os.path.basename(path), re.I)
	return m.group(1), m.group(2)

def patient_id(pair):
	return key(pair[0])[0]

def get_pairs(root):
	"""Match every MRI with its label by (patient, week). Returns [(img_path, lbl_path), ...]."""
	L = layout()
	imgs = glob.glob(os.path.join(root, L["mr"], "*.nii.gz"))
	lbls = {key(p): p for p in glob.glob(os.path.join(root, L["lbl"], "*.nii.gz"))}
	pairs = [(i, lbls[key(i)]) for i in sorted(imgs) if key(i) in lbls]
	assert pairs, f"No image/label pairs under {os.path.join(root, L['mr'])}"
	return pairs

def split_pairs(root, pairs, seed=42, frac=(0.7, 0.15, 0.15)):
	"""Patient-level split. Uses the course's 2D train/validate/test patients if available."""
	d2 = os.path.join(root, layout()["d2"])
	ids = {}
	for split, folder in [("train", "keras_slices_train"),
						  ("val", "keras_slices_validate"),
						  ("test", "keras_slices_test")]:
		files = glob.glob(os.path.join(d2, folder, "*.nii.gz"))
		ids[split] = {re.search(r"(\d+)", os.path.basename(f)).group(1) for f in files}

	if not all(ids.values()):          # 2D folders missing/empty -> random patient split
		print("WARNING: 2D split not found, using random patient split")
		pids = sorted({patient_id(p) for p in pairs})
		random.Random(seed).shuffle(pids)
		n = len(pids); a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
		ids = {"train": set(pids[:a]), "val": set(pids[a:b]), "test": set(pids[b:])}

	# leakage check: no patient in two splits
	assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
	return {s: [p for p in pairs if patient_id(p) in g] for s, g in ids.items()}

class HipMRI3D(Dataset):
	"""One (image, label) volume per item: image (1, D, H, W) float, label (D, H, W) long."""
	def __init__(self, pairs, scale=0.5):
		self.pairs, self.scale = pairs, scale

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, i):
		ip, lp = self.pairs[i]
		img = nib.load(ip).get_fdata(caching="unchanged")
		lbl = nib.load(lp).get_fdata(caching="unchanged")
		if img.ndim == 4: img = img[..., 0]
		if lbl.ndim == 4: lbl = lbl[..., 0]
		img = (img - img.mean()) / (img.std() + 1e-8)          # z-score per volume
		# files are (H, W, D); model wants (D, H, W); add batch+channel axes for interpolate
		img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None, None]
		lbl = torch.from_numpy(np.rint(lbl).astype(np.float32)).permute(2, 0, 1)[None, None]
		if self.scale != 1:
			img = F.interpolate(img, scale_factor=self.scale, mode="trilinear", align_corners=False)
			lbl = F.interpolate(lbl, scale_factor=self.scale, mode="nearest")  # keeps class IDs intact
		return img[0], lbl[0, 0].long()
[Hard Difficulty] Use downsampled HipMRI 3D data. Ref: [5]. 3D dilated
context aggregation; target Dice ≥ 0.7 on test set. Use
pyimgaug3d or PyTorch 3D transforms.

## Quick info

using a Context Aggregate Network (CAN) keeps every layer at full resolution, but
has a 'dilation' in the convolutional layers to increase the receptive field and
keep the memory somewhat manageable. The network will be made up of a daisy chain of
'dilated blocks', that each perform convolution, normalization and relu activation (leaky).
The dilation in each block grows, increasing the receptive field.

## Memory usage

Each layer in the CAN stores Width * D * H * W values.

## Artificial Intelligence Usage Disclosure

* I have used Claude
* It helped me

1. modules.py: Containing the source code of the components of your model. Each component must be
   implemented as a class or a function in PyTorch or TensorFlow/Keras. Your implementation of the
   model should not depend on NumPy (or other Python libraries not part of PyTorch / TensorFlow)
   in any way unless otherwise approved by the teaching staff.

2. dataset.py: Containing the data loader for loading and preprocessing your data, including leakagefree train/validation/test splitting functions and data augmentations.

3. train.py: Containing the source code for training, validating, testing, and saving your model.
   The model should be imported from modules.py and the data loader should be imported from
   dataset.py. Make sure to plot the losses and metrics during training.

4. predict.py: Showing example usage of your trained model. It should load your saved model
   weights, run inference on test cases, print out any results, and provide visualisations where applicable
   (e.g., prediction overlays, generated samples, heatmaps). NumPy is allowed in this script for
   visualization or loading data.

5. README.md: Sufficiently documenting your project, working principles, feasibility review, experiments,
   and instructions (see Section 5.3).

```python
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os, glob

def get_paths(root):
	imgs = sorted(glob.glob(os.path.join(root, "semantics_MRs*", "*.nii.gz")))
	lbls = sorted(glob.glob(os.path.join(root, "semantics_labels*", "*.nii.gz")))
	return imgs, lbls


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
	channels = np.unique(arr)
	res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
	for c in channels:
		c = int(c)
		res[..., c:c+1][arr == c] = 1
	return res

def load_data_2D(imageNames, normImage=False, categorical=False,
				 dtype=np.float32, getAffines=False, early_stop=False):
	'''
	Load medical image data from names, cases list provided into a list for each.

	This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

	normImage : bool (normalise the image 0.0-1.0)
	early_stop : Stop loading pre-maturely, leaves arrays mostly empty,
				 for quick loading and testing scripts.
	'''
	affines = []

	# get fixed size
	num = len(imageNames)
	first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
	if len(first_case.shape) == 3:
		first_case = first_case[:, :, 0]  # sometimes extra dims, remove
	if categorical:
		first_case = to_channels(first_case, dtype=dtype)
		rows, cols, channels = first_case.shape
		images = np.zeros((num, rows, cols, channels), dtype=dtype)
	else:
		rows, cols = first_case.shape
		images = np.zeros((num, rows, cols), dtype=dtype)

	for i, inName in enumerate(tqdm(imageNames)):
		niftiImage = nib.load(inName)
		inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
		affine = niftiImage.affine
		if len(inImage.shape) == 3:
			inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
		inImage = inImage.astype(dtype)
		if normImage:
			inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
		if categorical:
			inImage = to_channels(inImage, dtype=dtype)
			images[i, :, :, :] = inImage
		else:
			images[i, :, :] = inImage

		affines.append(affine)
		if i > 20 and early_stop:
			break

	if getAffines:
		return images, affines
	else:
		return images


def load_data_3D(imageNames, normImage=False, categorical=False,
				 dtype=np.float32, getAffines=False, orient=False,
				 early_stop=False):
	'''
	Load medical image data from names, cases list provided into a list for each.

	This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

	normImage : bool (normalise the image 0.0-1.0)
	orient : Apply orientation and resample image? Good for images with
			 large slice thickness or anisotropic resolution
	dtype : Type of the data. If dtype=np.uint8, it is assumed that the data is labels
	early_stop : Stop loading pre-maturely? Leaves arrays mostly empty,
				 for quick loading and testing scripts.
	'''
	affines = []

	interp = 'linear'
	if dtype == np.uint8:  # assume labels
		interp = 'nearest'

	# get fixed size
	num = len(imageNames)
	niftiImage = nib.load(imageNames[0])
	first_case = niftiImage.get_fdata(caching='unchanged')
	if len(first_case.shape) == 4:
		first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
	if categorical:
		first_case = to_channels(first_case, dtype=dtype)
		rows, cols, depth, channels = first_case.shape
		images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
	else:
		rows, cols, depth = first_case.shape
		images = np.zeros((num, rows, cols, depth), dtype=dtype)

	for i, inName in enumerate(tqdm(imageNames)):
		niftiImage = nib.load(inName)
		inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
		affine = niftiImage.affine
		if len(inImage.shape) == 4:
			inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
		inImage = inImage[:, :, :depth]  # clip slices
		inImage = inImage.astype(dtype)
		if normImage:
			inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
		if categorical:
			inImage = to_channels(inImage, dtype=dtype)
			images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :] = inImage
		else:
			images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage

		affines.append(affine)
		if i > 20 and early_stop:
			break

	if getAffines:
		return images, affines
	else:
		return images
```

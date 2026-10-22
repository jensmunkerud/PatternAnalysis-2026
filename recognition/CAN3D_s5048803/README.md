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


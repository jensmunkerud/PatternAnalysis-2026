[Hard Difficulty] Use downsampled HipMRI 3D data. Ref: [5]. 3D dilated
context aggregation; target Dice ≥ 0.7 on test set. Use
pyimgaug3d or PyTorch 3D transforms.


## Quick info
using a Context Aggregate Network (CAN) keeps every layer at full resolution, but
has a 'dilation' in the convolutional layers to increase the receptive field and
keep the memory somewhat manageable. The network will be made up of a daisy chain of
'dilated blocks', that each perform convolution, normalization and relu activation (leaky).
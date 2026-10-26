# CAN 3D Prostate segmentation model

## Notes
* The loss function uses Cross-Entropy for optimal discrete segmentation performance
* Dice-score is calculated on foreground classes to avoid falsely high model performance
* 3D Augmentation with PyTorch is used for the 3D dataset to increase variation and avoid overfitting
* Uses one-hot encoding to avoid falsely ranking the order of different classes

perhaps put each note under its relevant section:

DATASET:
* Pads necessary slices to every slice is 128x64 (some are 128x144)



# Feasibility report

Author: s5048803



## User Need, Scope & Acceptance Criteria

This machine learning model is intended to give certified personnell a rough segmentation of the prostate in 2d and 3d MRI data. It could cut down time and cost of such work. Because of this its important that the model is fast and light to run. A baseline CAN2D model will work on individual slices while a CAN3D version will work on fully 3d MRI data. They will then be compared.
The a100 gpu used for training have about 40GB ram. Running inference on a CAN model is usually lighter than a U-NET, meaning it's easier to run on end-user devices, making the model practical for real-world use.

I'll specifically inspect these boundary failure cases:

* Adjoining objects around the prostate
* Stray floating artifacts (separated islands)
* Overcoverage / undercoverage
* Jagged edges

The acceptance criteria are:

* Overall Dice accuracy must be >= 0.75 for both 2D and 3D
* Peak VRAM must be < 40GB to fit on the A100 during training
* Peak inference memory <= 8GB preferably, absolute max 16GB to run on end-user hardware.
* Difference between train and validation Dice <= 0.1 to prevent overfitting on such a small dataset (preferably <= 0.05)
* Inference runs in <10 seconds on M2 mac

## Model Choice & Course Concepts

I'll be developing the CAN3D from [hard] difficulty, and compare it to a similar CAN2D model. The core architecture between the two will be the same, though the 3D version might need some different features later on in training to keep computing resources under control, and the dataloader will be different. The Context Aggregation Network will be a streamlined process through 'Dilation blocks'. Each of these blocks will perform convolution with dilation, normalization and then activation. Different dilations are used for each block to balance fine detail and wide context. An important feature with these dilation blocks is that the output dimension is the same as the input dimension. Lastly, a convolutional layer is used to group the internal featuremaps into the 6 different classes: Background, skin, skeleton, bladder, rectum and prostate.
LeakyRelu is used instead of regular Relu to avoid dead units. This is especially important with an architecture like this, where we have no skip connections and every gradient must pass through the whole network.
The train, test and validation datasets will be kept as similar as possible between the 2D and 3D models to give them as fair comparison as possible. The CAN3D: FAST 3D MEDICAL IMAGE SEGMENTATION VIA COMPACT CONTEXT AGGREGATION paper by Wei Dai and Shakes etc. will be useful.

## Preliminary Feasibility Evidence

The data: 38 patients. The 2D dataset is bigger because it has all the different slices of the patients while the 3D model treats the whole scan as a single input. The previously mentioned acceptance criteria is therefore very important to meet, since this is a small dataset.
The memory usage of the 3D model: 
Kernel size = 3 
Network width = 24 
float32 
Assuming a 2x downsampling (256x256x128 -> 128x128x64), the amount of parameters are:

3^3 * 1  * 24 + 24 = 672 parameters for initial block 
3^3 * 24 * 24 + 24 = 15 576 parameters per central block (4) 
1   * 24 * 6  + 6  = 150 parameters for final layer 
Total = 63 126 parameters (252.5 KB)
The can3d_best.pt is 258KB, almost exactly this, and does not change with changing input resolution.

Activations:
24 * 64 * 128 * 128 * 4 bytes $\approx$ 100MB each layer 
Each block has 3 layers, giving a rough model memory estimate of 1.6GB -> 2GB. 
When running inference, about only one layer is active at once leading to as low as 1/16th -> 1/8th memory usage of training. The following tests ran with a 2x downsampling of the data, and shows we are well within the memory limit.
This gives headroom to increase the resolution of the data input aswell as the amount of layers in the model, if it performs better that way.

I ran a mock training of the CAN3D model with 15 epochs, and got the following from slurm:

| TYPE  | EPOCHS | TIME     | MEMORY USAGE   |
| ----- | ------ | -------- | -------------- |
| CAN3D | 15     | 00:07:43 | 3059388K = 3GB |

Results from running the CAN3D model on 4 samples. We clearly see the low resolution of the data, but the model shows
promising results considering this took less than 10min to train in only 15 epochs.

![alt text](outputs/can3d_predictions.png)

## Risks, Budget & Fallback

With this being a Context Aggregation Network (CAN), it keeps the input data at the same, high resolution all the way through the network. During training, this usually means it will use more RAM during training compared to U-NET for instance. It will also be challenging to balance its broad spacial awareness vs the fine detail accuracy. This will come down to tweaking the dilations for the different layers.

Furthermore, since the CAN2D and CAN3D architectures are very similar in design, it is paramount that they both work for this paper.
The initial training shows the architecture works, so if the data results are not sufficient a fallback would be to make it run on the OASIS brain dataset instead.
15 epochs took around 8 minutes, and i expect the full run to be around 100 epochs. With some headroom for the downsampling to be lessened I'm expecting training to take 1-2 hours on the A100 GPU. We have ~26 hours allocated on a dedicated lab computer.

The next experiment will be to compare the 2d and 3d results with eachother, and start to look at data proving whether or not the additional dimension is worth it. Also increasing the sample resolution while still staying inside the memory limits will be important to balance for the models usability and quality of output. I'd also like to generate a 3d plot of the prostate organ and compare it with the ground truth.

# FEEDBACK

* Plot per-slice difference between CAN2D and CAN3D segmentation (total area).
* Consider what the tradeoff between an overall low, but existing over/underfitting is vs a big spiked boundary.
* /\ /\ if this above is not good enough, you must find a good way to compare if the 3d version performs better than the 2d
* Ensure the test data that the predict runs on uses the exact same slices, otherwise the result could be misleading.
* Keep all the classes
* Calculate dice-score on a per-class basis
* The model predicts 6 classes, and is not biased towards any of them. The models performance will thus be reflected in the segmentation of every class.
  This means that the error between the ground truth and estimated segmentation for ALL classes is taken into account when evaluating each slice.

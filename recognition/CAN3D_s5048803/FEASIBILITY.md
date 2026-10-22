# Feasibility report

## User Need, Scope & Acceptance Criteria
This machine learning model is intended to give certified personnell a rough segmentation of the prostate in 2d and 3d MRI data. It could cut down time and cost of such work. Because of this its important that the model is fast and light to run.
The a100 gpu used for training have about 40GB ram. Running inference on a CAN model is usually lighter than a U-NET, meaning it's easier to run on end-user devices, making the model practical for real-world use.
The acceptance criteria are:
* Overall Dice accuracy must be >= 0.75 for both 2D and 3D
* Peak VRAM must be < 40GB to fit on the A100 during training
* Peak inference memory <= 8GB preferably, absolute max 16GB to run on end-user hardware.
* Difference between train and validation Dice <= 0.1 to prevent overfitting on such a small dataset (preferably <= 0.05)
* Inference runs in <10 seconds on M2 mac


## Model Choice & Course Concepts
I'll be developing the CAN2D from [normal] difficulty first before comparing it to the fully-fledged CAN3D [hard]. The core architecture between the two will be the same, though the 3D version might need some different features later on in training to keep computing resources under control, and the dataloader will be different. The Context Aggregation Network will be a streamlined process through 'Dilation blocks'. Each of these blocks will perform convolution with dilation, normalization and then activation. Different dilations are used for each block to balance fine detail and wide context. An important feature with these dilation blocks is that the output dimension is the same as the input dimension. Lastly, a convolutional layer is used to group the internal featuremaps into the 6 different classes: Background, skin, skeleton, bladder, rectum and prostate.
LeakyRelu is used instead of regular Relu to avoid dead units. This is especially important with an architecture like this, where we have no skip connections and every gradient must pass through the whole network.
The train, test and validation datasets will be kept as similar as possible between the 2D and 3D models to give them as fair comparison as possible. The CAN3D: FAST 3D MEDICAL IMAGE SEGMENTATION VIA COMPACT CONTEXT AGGREGATION paper by Wei Dai and Shakes etc. will be useful.


## Preliminary Feasibility Evidence
The data: 38 patients . The 2D dataset is 
The memory usage of the 3D model:
Kernel size = 3
Network width = 24
float32
Assuming a 2x downsampling (256x256x128 -> 128x128x64)the amount of parameters are:
128x128x64 = 1 048 576

3^3 * 1  * 24 + 24 = 672 parameters for initial block
3^3 * 24 * 24 + 24 = 15 576 parameters per central block (4)
1   * 24 * 6  + 6  = 150 parameters for final layer
Total = 63 126 parameters

Activations:
24 * 64 * 128 * 128 * 4 bytes = 100MB each layer
Each block has 3 layers, giving a rough model memory estimate of 1.6GB -> 2GB
The following tests ran with a 2x downsampling of the data, and shows we are well within the memory limit.

I ran a mock training of the CAN3D model with 15 epochs, and got the following from slurm:
| TYPE  | EPOCHS | TIME     | MEMORY USAGE          |
|-------|--------|----------|-----------------------|
| CAN3D | 15     | 00:07:43 | 3059388K = 3GB        |
| CAN2D | 

Results from running the CAN3D model on 4 samples. We clearly see the low resolution of the data, but the model shows
promising results considering this took less than 10min to train in only 15 epochs.

![alt text](outputs/can3d_predictions.png)


## Risks, Budget & Fallback
With this being a Context Aggregation Network (CAN), it keeps the input data at the same, high resolution all the way through the network. This means it will have a relatively high amount of weights and biases compared to a more spacially efficient architecture, like U-NET. This leads to more RAM usage and I've got to carefully manage th.

Furthermore, since the CAN2D and CAN3D architectures are very close.

The next experiment will be to compare the 2d and 3d results with eachother, and start to look at data proving whether or not the additional dimension is worth it. Also increasing the sample resolution while still staying inside the memory limits will be important to balance for the models usability and quality of output. I'd also like to generate a 3d plot of the prostate organ and compare it with the ground truth.
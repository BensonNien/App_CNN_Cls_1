
# 
## CNN Classification with training and inferance using pure C++ implement

This program is the prototype with tiny architecture. You can figure out how mechanisms working in the detail.
CPU version is ready. GPU (CUDA) version is coming soon.

## Instructions for Code:
### Requirements

IDE: VS2019

Library: OpenCV >= 3.4.17 (images read only)

### Description

Simple training dataset is included in the folder named "Pedestrian_TrainingDataset_PNG" with balance negtive and position samples.
The trained model's weights are stored in the dirs of "./data/kernel_weight/" and "./data/bias/".
Load model's weights using LoadParas() for pretraining or inference.

### Training and Inference

Please refer to comments in main().

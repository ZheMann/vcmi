# Video Camera Model Identification (Digital Forensics)
This repository consists of code to perform the task of video camera model identification. Experiments are conducted by using data of the [VISION](https://lesc.dinfo.unifi.it/it/node/203) dataset.

## ConstrainedNet vs. ConvNet
In constrained_net.py the parameter self.constrained_net can be set to False to remove the constrained convolutional layer. 

## Creation of datasets
To create the datasets, we assume the VISION dataset is available in its original structure. Obviously, we only need the videos for each device so images may be skipped. 

1. Execute frame_extractor.py by setting param --input_dir="/path/to/VISION dataset". This script iterates over each device and extracts N frames from each video. A separate directory is created for every video, consisting of N frames. 


## Experiment 1. Frames
Video classification by providing entire frames to the ConstrainedNet. Frames are resized to size of 480x800.

### Frame dataset
To create the frames dataset:
1. Run create_main_dataset.py
2. Run create_train_test_dataset

## Experiment 2. Patches
Video classification by providing patches of size 128x128 to the ConstrainedNet. The input size of the ConstrainedNet is also set to 128x128 so the patches are not resized. 

### Frame dataset




## Training


## Predicting

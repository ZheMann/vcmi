# Video Camera Model Identification (Digital Forensics)
This repository consists of code to perform the task of video camera model identification. Experiments are conducted by using data of the [VISION](https://lesc.dinfo.unifi.it/it/node/203) dataset.

## Creation of datasets
To create the datasets, we assume the VISION dataset (only videos) is available in the original structure:
```
VISION  
  |  
  |--- D01  
  |     |  
  |      --- videos  
  |             |  
  |              --- flat  
  |                   |  
  |                    --- flat_video1  
  |                    --- flat_video2  
  |              --- flatWA  
  |                   |  
  |                    --- flatWA_video1  
  |                    --- flatWA_video2  
  |              --- ...  
  |              --- outdoorYT  
   --- D02  
   --- ...  
   --- D35  
   
```

### Frame dataset
In the following steps we refer to the VISION dataset as the VISION video dataset. 

1. Execute `frame_extractor.py` and set params `--input_dir="/path/to/VISION video dataset"` and `--output_dir=/path/to/VISION frame dataset`. This script creates a new directory (i.e. `/path/to/VISION frame dataset`) with the same tree-like structure as presented in the VISION video dataset. This script iterates over each device in the `VISION video dataset`, and extracts `N` frames from each video. A separate directory is created for every video. To change the number of frames extracte from a video, change param `--frames_to_save_per_video`.

2. Execute `create_main_test.py`. Set property `VISION_DATASET_DIR` (line 54) to the path to `/path/to/VISION video dataset` and set `VISION_FRAMES_DIR` (line 55) to `/path/to/VISION frame dataset`. This script creates a new dataset for 28 devices (hard-defined in script itself) with only valid videos. A video is considered to be valid when both WhatsApp and YouTube versions are available for the native/original video. 

The structure of the main dataset is as follows:
```
DATASET
  |
   --- D01
  |     |
  |      --- Video1
  |             |
  |              --- Video1_frame1.jpeg
  |              --- Video1_frame2.jpeg
  |      --- VideoWA1
  |      --- VideoYT1
   --- D02
```

This script is not optimised as it copies frames instead of using symlinks.

3. Execute `create_train_test_set.py` to create the train and test based on the dataset created in step 2. This script randomly selects 7 train and 6 test videos per device, including the social versions.  

### Patch dataset
The patch train and test set are created by copying the structure of the train and test set of the frame dataset. The following scripts can be executed to create and balance the patch train and test set:

1. `patch_extractor_train.py` to create a new (unbalanced) train set of patches of size 128x128.
2. `patch_extractor_test.py` to create a new (unbalanced) test set of patches of size 128x128.
3. `patch_balancer_train.py` to balance the train set by ensuring that each video is represented by same number of frames (number of patches may differ though).
4. `patch_balancer_test.py` same as step 3 but for the test set.

## Training
Script `constrained_net/train/train.py` can be executed to train the ConstraindNet. I provided bash files to train the ConstrainedNet in `constrained_net/bash/*`.

## Evaluating / Predicting
Script `constrained_net/predict/predict_flow.py` can be executed to automatically generate predictions on frame and video level. Param `--input_dir` should point to a directory consisting of models (.h5 file). In my case, I saved a model after every epoch. The script generates frame and video predictions for each model available in the input directory. If you only want specific models to be evaluated, use param `--models` to specifiy the filenames of the models (separated by a comma). 

Script `predict_flow.py` involves many steps, which I will further explain. 

To create frame and video predictions, we first create two csv-files for every model in the input directory:

1. In `predict_frames.py`, frames are predicted by the ConstrainedNet and subsequently saved to a csv-file.
2. In `predict_videos.py`, the frame prediction csv-file (produced in step 1.) is loaded and used to predict videos by the majority vote. Video predictions are saved in a separate csv-file.

This results in `K` frame prediction csv-files and `K` video prediction csv-files where `K` represents the number of models in the input directory. To visualize the prediction results, we have to calculate the statistics for every frame and video csv-file, which is done as follows:

3. In `frame_prediction_statistics.py`, frame statistics (averages per scenarios, platforms, etc.) are generated for every frame prediction csv-file.
4. In `video_prediction_statistics.py`, video statistics (averages per scenarios, platforms, etc.) are generated for every frame prediction csv-file.

This results in 1 frame statistics csv-file and 1 video statistics csv-file. The statistics files have as many rows as there are models available in the input directory. Lastly, these files are used to visualize the results in:

5. `frame_prediction_visualization.py` to visualize frame prediction results.
6. `video_prediction_visualization.py` to visualize video prediction results.

## ConstrainedNet vs. ConvNet
In constrained_net.py the parameter self.constrained_net can be set to False to remove the constrained convolutional layer. 


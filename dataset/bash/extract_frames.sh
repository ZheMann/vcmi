#!/bin/bash

#SBATCH --job-name=balance_train_patch
#SBATCH --time=04:00:00
#SBATCH --mem=40GB

pip install opencv-python --user

# echo running_the_flow
python3 ../frames/frame_extractor/frame_extractor.py --input_dir="VISION dataset" --output_dir="VISION frames" --frames_to_save_per_video=200
# echo completed_the_job
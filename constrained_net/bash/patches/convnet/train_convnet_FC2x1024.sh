#!/bin/bash

#SBATCH --job-name=Exp.IV_CNN_FC2x1024_28D_patches
#SBATCH --time=16:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

# echo starting_jobscript
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4
module load Pillow/6.2.1-GCCcore-8.3.0

# echo running_the_flow
python3 ../../train/train.py --dataset="Exp.2/balanced_patches_ds_28D/" --epochs=30 --batch_size=64 --model_name="ConvNet_FC2x1024_28D_patches" --height=128 --width=128
# To continue training from a previous model add parameter --model_path with the value pointing to the particular *.h5 model

# echo completed_the_job
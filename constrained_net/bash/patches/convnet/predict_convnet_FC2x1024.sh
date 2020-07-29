#!/bin/bash

#SBATCH --job-name=predict-constrained-single
#SBATCH --time=03:00:00
#SBATCH --mem=30000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

# echo starting_jobscript
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4
module load Pillow/6.2.1-GCCcore-8.3.0

# We need to install seaborn manually
pip install seaborn --user

# echo running_the_flow
python3 ../../../predict/predict_flow.py --input_dir="/data/s3523799/vbdi/models/ConvNet_FC2x1024_28D_patches/" --dataset="Exp.2/balanced_patches_ds_28D/" --batch_size=64 --height=128 --width=128
# echo completed_the_job
#!/bin/bash

#SBATCH --job-name=predict-FC1_1024-FC2_1024
#SBATCH --time=14:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

# echo starting_jobscript
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-foss-2019b-Python-3.7.4
module load Pillow/6.2.1-GCCcore-8.3.0

# We need to install seaborn manually
pip install seaborn --user

# echo running_the_flow
python3 ../../../predict/predict_flow.py --dataset="Exp.1/28D_ds/" --batch_size=64 --constrained=1 --input_dir="/data/s3523799/vbdi/models/ConstrainedNet_FC2x1024_28D/"
# echo completed_the_job
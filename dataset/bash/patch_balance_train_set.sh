#!/bin/bash

#SBATCH --job-name=balance_train_patch
#SBATCH --time=04:00:00
#SBATCH --mem=40GB

pip install opencv-python --user

# echo running_the_flow
python3 ../patches/patch_balancer_train.py
# echo completed_the_job
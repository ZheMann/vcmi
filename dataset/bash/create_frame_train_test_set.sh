#!/bin/bash

#SBATCH --job-name=extract_train_patch
#SBATCH --time=01:00:00
#SBATCH --mem=30GB

# echo running_the_flow
python3 ../frames/2.create_train_test_set.py
# echo completed_the_job
#!/bin/bash
#SBATCH --job-name=Unet_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=basic, gpu
#SBATCH --mem=200GB  # Increased memory
#SBATCH --time=5-12:00:00
#SBATCH --gres=shard:32
#SBATCH --output=log_unet_train.out
#SBATCH --error=log_unet_train.err

# Activate your conda environment
source activate unet_server

# Navigate to the directory of the script
cd $(dirname "$0")

# Run the Python script
python train.py

# End of the script

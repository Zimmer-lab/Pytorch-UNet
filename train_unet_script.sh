#!/bin/bash
#SBATCH --job-name=Unet_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=basic,gpu
#SBATCH --mem=200GB  # Increased memory
#SBATCH --time=5-12:00:00
#SBATCH --gres=shard:32
#SBATCH --output=log_unet_train.out
#SBATCH --error=log_unet_train.err

# Print the start of the script
echo "Starting the script..."

# Activate your conda environment
echo "Activating conda environment..."
source activate unet_server

# Check which conda environment is activated
echo "Currently activated environment:"
conda info --envs

# Confirm current working directory
echo "Current directory: $(pwd)"

# Run the Python script
echo "Starting Python script..."
python /lisc/scratch/neurobiology/zimmer/schaar/code/github/unet_modified/Pytorch-UNet/train.py

# End of the script
echo "Script ended."

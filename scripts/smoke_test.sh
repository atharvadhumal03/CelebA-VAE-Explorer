#!/bin/bash
#SBATCH --job-name=latentlens-smoke
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-smoke-%j.out
#SBATCH --error=logs/slurm-smoke-%j.err

# Always pull latest code before running
git pull

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latentlens

echo "Running smoke test (1000 images, 2 epochs)..."
python -m src.train --config configs/default.yaml --smoke-test

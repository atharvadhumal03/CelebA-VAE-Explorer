#!/bin/bash
#SBATCH --job-name=latentlens-train
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Always pull latest code before running
git pull

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latentlens

# Auto-resume from latest checkpoint if one exists
LATEST_CKPT=$(ls -t outputs/checkpoints/checkpoint_epoch*.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    python -m src.train --config configs/default.yaml --resume "$LATEST_CKPT"
else
    echo "Starting fresh training run"
    python -m src.train --config configs/default.yaml
fi

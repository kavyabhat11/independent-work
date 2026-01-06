#!/bin/bash
#SBATCH --job-name=mozart-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j-ft.out

set -eo pipefail

cd "/home/kb9520/KAVYA IW FINAL"
source ~/.bashrc
conda activate chordgpu

# useful debug info
nvidia-smi || true
python -c "import torch; print('cuda?', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python finetune_from_checkpoint.py

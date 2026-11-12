#!/bin/bash
#SBATCH --job-name=beta_tuning
#SBATCH --output=tuning_run_%j.log
#SBATCH --error=tuning_run_%j.err
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00

set -e
export WANDB_MODE=disabled
source /work/axl220095/envs/beta_env/bin/activate
cd /work/axl220095/BETA

DATA_R="/work/axl220095/datasets/imagenet-r"
STEPS=100

python main.py --algorithm beta --corruption rendition --data_rendition "$DATA_R" --steps $STEPS > out_beta_default.txt 2>&1
python main.py --algorithm beta --corruption rendition --data_rendition "$DATA_R" --steps $STEPS --alpha 0.2 --margin_e0 0.95 --bvr_lr 0.0005 > out_beta_tuned.txt 2>&1

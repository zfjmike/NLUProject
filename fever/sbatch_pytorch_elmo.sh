#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=24GB
#SBATCH --job-name=esim+elmo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fz758+monitor@nyu.edu
#SBATCH --output=hpc_outputs/slurm_%j.out
#SBATCH --error=hpc_outputs/slurm_%j.err
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ucl
cd /scratch/fz758/ucl/fever

python pipeline.py --config configs/pytorch_esim_elmo.json --overwrite --model pytorch_esim_elmo
#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=esimsa_elmo_10k_n5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fz758+monitor@nyu.edu
#SBATCH --output=hpc_outputs/slurm_%j.out
#SBATCH --error=hpc_outputs/slurm_%j.err
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ucl
cd /scratch/fz758/ucl/fever

python pipeline.py --config configs/esimsa_elmo_10k_n5.json --overwrite --model esimsa_elmo_10k_n5
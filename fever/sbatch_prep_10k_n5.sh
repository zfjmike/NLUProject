#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=24GB
#SBATCH --job-name=prep_10k_n5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fz758+monitor@nyu.edu
#SBATCH --output=hpc_outputs/slurm_%j.out
#SBATCH --error=hpc_outputs/slurm_%j.err

source ~/.bashrc
conda activate ucl
cd /scratch/fz758/ucl/fever

python pipeline.py --config configs/depsa_10k_n5.json --overwrite --model depsa_10k_n5
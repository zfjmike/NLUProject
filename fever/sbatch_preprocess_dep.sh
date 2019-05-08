#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mem=48GB
#SBATCH --job-name=preprocess_dep
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fz758+monitor@nyu.edu
#SBATCH --output=hpc_outputs/slurm_%j.out
#SBATCH --error=hpc_outputs/slurm_%j.err

source ~/.bashrc
conda activate ucl
cd /scratch/fz758/ucl/fever

python pipeline.py --config configs/pytorch_depsa_esim.json --overwrite --model pytorch_depsa_esim
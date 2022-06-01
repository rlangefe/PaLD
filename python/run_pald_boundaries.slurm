#!/bin/bash
#SBATCH --job-name="boundaries"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=10GB
#SBATCH --account=classes
#SBATCH --mail-user=langrc18@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=boundaries-%j.o
#SBATCH --error=boundaries-%j.e

module load rhel7/gcc/6.2.0 rhel7/gcc/6.2.0-libs
module load rhel7/gpu/cuda/10.2

python -u plot_boundaries.py
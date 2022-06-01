#!/bin/bash
#SBATCH --job-name="PaLD Sim R"
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cascade
#SBATCH --array=1-1000%10
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/pald-sim-%j-%a.o

MAINDIR=/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/R

module load rhel7/R/4.0.2

cd $MAINDIR

Rscript runSim.R $((CURR_ITER * 1000 + SLURM_ARRAY_TASK_ID))

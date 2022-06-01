#!/bin/bash
#SBATCH --job-name="PaLD Run"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=2
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=pald-run-%j.o
#SBATCH --error=pald-run-%j.e
#SBATCH --mail-user=langrc18@wfu.edu

module load rhel7/R/4.0.2

Rscript pald.R

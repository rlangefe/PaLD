#!/bin/bash
#SBATCH --job-name="PaLD Matlab Muday"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=langrc18@wfu.edu
#SBATCH --output=pald-muday-matlab-%j.o
#SBATCH --error=pald-muday-matlab-%j.e

cd /deac/csc/classes/csc391_2/csc391/langrc18/PaLD/matlab

module load rhel7/matlab/2020a

MFILE=pald.m
OPTIONS="-nodesktop -nosplash"

matlab -r "run('$MFILE');" $OPTIONS
#!/bin/bash
#SBATCH --job-name="PaLD Sim Matlab"
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=20
#SBATCH --time=07-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=langrc18@wfu.edu
#SBATCH --output=pald-sim-%j.o
#SBATCH --error=pald-sim-%j.e

module load rhel7/matlab/2020a

TOP="/deac/csc/classes/csc391_2/csc391/langrc18/PaLD"
MATLAB_DIR=${TOP}/matlab

cd $MATLAB_DIR
echo "Running PaLD"
MFILE=simRun.m
OPTIONS="-nodesktop -nosplash -nojvm -batch"

matlab $OPTIONS "run('$MFILE');"

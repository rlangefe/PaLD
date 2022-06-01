#!/bin/bash
#SBATCH --job-name="PaLD Sim Matlab"
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --tasks-per-node=10
#SBATCH --time=07-00:00:00
#SBATCH --constraint=cascade
#SBATCH --array=1-1000
#SBATCH --output=logs/pald-sim-%j.o

module load rhel7/matlab/2020a

TOP="/deac/csc/classes/csc391_2/csc391/langrc18/PaLD"
MATLAB_DIR=${TOP}/matlab
export SCRATCH=/scratch

cd $MATLAB_DIR
echo "Running PaLD"
MFILE=simRun_parallel.m
OPTIONS="-nodesktop -nosplash -nojvm -batch"

matlab $OPTIONS "curr_arr=$SLURM_ARRAY_TASK_ID; run('$MFILE');"

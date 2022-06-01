#!/bin/bash
#SBATCH --job-name="PaLD Sim Matlab"
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --tasks-per-node=44
#SBATCH --time=01-00:00:00
#SBATCH --constraint=cascade
#SBATCH --output=scaling_logs/pald-sim-%j.o

module load rhel7/matlab/2020a

TOP="/deac/csc/classes/csc391_2/csc391/langrc18/PaLD"
MATLAB_DIR=${TOP}/matlab
export SCRATCH=/scratch

cd $MATLAB_DIR
echo "Running PaLD"
echo "Procs: $SLURM_NTASKS_PER_NODE"
MFILE=simRun_scaling.m
OPTIONS="-nodesktop -nosplash -nojvm -batch"

matlab $OPTIONS "curr_procs=$SLURM_NTASKS_PER_NODE; run('$MFILE');"

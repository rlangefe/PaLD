#!/bin/bash
#SBATCH --job-name="PaLD Sim Python"
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --tasks-per-node=1
#SBATCH --time=01-00:00:00
#SBATCH --constraint=cascade
#SBATCH --array=1-1000
#SBATCH --output=logs/pald-sim-%j.o

TOP="/deac/csc/classes/csc391_2/csc391/langrc18/PaLD"
PYTHON_DIR=${TOP}/python
export SCRATCH=/scratch

cd $PYTHON_DIR
python simRun_cpu_parallel.py

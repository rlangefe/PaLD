#!/bin/bash
#SBATCH --job-name="PaLD Sim"
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --time=07-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=langrc18@wfu.edu
#SBATCH --output=pald-sim-cpu-%j.o
#SBATCH --error=pald-sim-cpu-%j.e

module load rhel7/gcc/6.2.0 rhel7/gcc/6.2.0-libs
module load rhel7/gpu/cuda/10.2

TOP="/deac/csc/classes/csc391_2/csc391/langrc18/PaLD"
SIM_DATA=${TOP}/data/sim
PYTHON_DIR=${TOP}/python

cd ${PYTHON_DIR}

python -u simRun_cpu.py -r 0 -o full_run_results_cpu.csv

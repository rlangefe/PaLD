#!/bin/bash
#SBATCH --job-name="PaLD Sim"
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=20
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=langrc18@wfu.edu
#SBATCH --output=pald-sim-%j.o
#SBATCH --error=pald-sim-%j.e

module load rhel7/matlab/2020a

TOP="/deac/csc/classes/csc391_2/csc391/langrc18/PaLD"
SIM_DATA=${TOP}/data/sim
MATLAB_DIR=${TOP}/matlab


for N in {0..3840}
do
    time
    SECONDS=0
    cd $SIM_DATA
    echo "Running ${N}"
    echo "Building Datasets"
    python ${TOP}/python/genDataset.py -i $N -f ${MATLAB_DIR}/info.csv
    python ${MATLAB_DIR}/combineSim.py -f1 "${SIM_DATA}/data_g1.csv" -f2 "${SIM_DATA}/data_g2.csv" -o "${SIM_DATA}/data.csv" -g "${SIM_DATA}/groups.csv"
    python ${MATLAB_DIR}/combineSim.py -f1 "${SIM_DATA}/test_g1.csv" -f2 "${SIM_DATA}/test_g2.csv" -o "${SIM_DATA}/test_data.csv" -g "${SIM_DATA}/test_groups.csv"

    cd $MATLAB_DIR
    echo "Running PaLD"
    MFILE=pald_and_predict.m
    OPTIONS="-nodesktop -nosplash -nojvm -batch"

    matlab $OPTIONS "run('$MFILE');"

    echo "Predicting"
    python prediction_methods.py -d ${SIM_DATA}
    echo "Run ${N} took ${SECONDS} seconds"
    echo
    echo
done

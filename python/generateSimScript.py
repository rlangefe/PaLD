import os
import argparse

if __name__ == '__main__':
    dist_list = ['exponential']*9
    dim_list = [1,2,3,4,5,10,100,500,2000]
    std_list = [1]*9
    mean_list = [1]*9

    option_list = zip(dist_list, dim_list, mean_list, std_list)

    i=0

    for dist, dim, mean, std in option_list:
        curr_file = '../bash/sim_scripts/' + str(dist) + '_' + str(dim) + '_sim.slurm'

        script='''#!/bin/bash
#SBATCH --job-name="PaLD ''' + str(dist) + ' ' + str(dim) + '''"
#SBATCH --partition=gpu
#SBATCH --nodelist=usb-gpu-03
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=1
#SBATCH --time=05-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=pald-sim-''' + str(dist) + '-' + str(dim) + '''-%j.o
#SBATCH --error=pald-sim-''' + str(dist) + '-' + str(dim) + '''-%j.e

module load rhel7/gpu/cuda/10.2 rhel7/gcc/7.5.0

export CUDA_VISIBLE_DEVICES=''' + str((i%5)+1) + '''

TOP="/deac/csc/classes/csc391_2/csc391/langrc18/PaLD"
PYTH_DIR="${TOP}/python"
CUDA_DIR="${TOP}/cuda/pald"

DIM=''' + str(dim) + '''
DISTRIBUTION="''' + str(dist) + '''"

CURR_SIM="${DISTRIBUTION}-${DIM}"

SIM_DIR="${TOP}/simulations/${CURR_SIM}"

mkdir -p ${SIM_DIR}

cd ${SIM_DIR}

for n in {100..10000..100}
do
    echo "Simulation size ${n}"
    for i in {1..15}
    do
        echo "Run ${i}"
        python ${PYTH_DIR}/simGen.py -n $n -d $DIM -m ''' + str(mean) + ''' -s ''' + str(std) + ''' -o "${SIM_DIR}/generated.csv" --dist ${DISTRIBUTION}

        ${CUDA_DIR}/main.x $n "${SIM_DIR}/generated.csv" 1 0.5 0 20 $DIM > /dev/null

        python ${PYTH_DIR}/convert.py -r rows.dat -c cols.dat -v values.dat -o "${SIM_DIR}/results.csv"

        python ${PYTH_DIR}/simCollect.py -n $n -d $DIM -m ''' + str(mean) + ''' -s ''' + str(std) + ''' -i "${SIM_DIR}/results.csv" -o "${SIM_DIR}/sim_record.csv"
    done
    echo
done

n=12000
echo "Simulation size ${n}"
for i in {1..15}
do
    echo "Run ${i}"
    python ${PYTH_DIR}/simGen.py -n $n -d $DIM -m ''' + str(mean) + ''' -s ''' + str(std) + ''' -o "${SIM_DIR}/generated.csv" --dist ${DISTRIBUTION}

    ${CUDA_DIR}/main.x $n "${SIM_DIR}/generated.csv" 1 0.5 0 20 $DIM > "${SIM_DIR}/temp_output.txt"

    python ${PYTH_DIR}/convert.py -r rows.dat -c cols.dat -v values.dat -o "${SIM_DIR}/results.csv"

    python ${PYTH_DIR}/simCollect.py -n $n -d $DIM -m ''' + str(mean) + ''' -s ''' + str(std) + ''' -i "${SIM_DIR}/results.csv" -o "${SIM_DIR}/sim_record.csv"
done
echo

n=20000
echo "Simulation size ${n}"
for i in {1..15}
do
    echo "Run ${i}"
    python ${PYTH_DIR}/simGen.py -n $n -d $DIM -m ''' + str(mean) + ''' -s ''' + str(std) + ''' -o "${SIM_DIR}/generated.csv" --dist ${DISTRIBUTION}

    ${CUDA_DIR}/main.x $n "${SIM_DIR}/generated.csv" 1 0.5 0 20 $DIM > "${SIM_DIR}/temp_output.txt"

    python ${PYTH_DIR}/convert.py -r rows.dat -c cols.dat -v values.dat -o "${SIM_DIR}/results.csv"

    python ${PYTH_DIR}/simCollect.py -n $n -d $DIM -m ''' + str(mean) + ''' -s ''' + str(std) + ''' -i "${SIM_DIR}/results.csv" -o "${SIM_DIR}/sim_record.csv"
done
echo

n=40000
echo "Simulation size ${n}"
for i in {1..15}
do
    echo "Run ${i}"
    python ${PYTH_DIR}/simGen.py -n $n -d $DIM -m ''' + str(mean) + ''' -s ''' + str(std) + ''' -o "${SIM_DIR}/generated.csv" --dist ${DISTRIBUTION}

    ${CUDA_DIR}/main.x $n "${SIM_DIR}/generated.csv" 1 0.5 0 20 $DIM > "${SIM_DIR}/temp_output.txt"

    python ${PYTH_DIR}/convert.py -r rows.dat -c cols.dat -v values.dat -o "${SIM_DIR}/results.csv"

    python ${PYTH_DIR}/simCollect.py -n $n -d $DIM -m ''' + str(mean) + ''' -s ''' + str(std) + ''' -i "${SIM_DIR}/results.csv" -o "${SIM_DIR}/sim_record.csv"
done
echo

python ${PYTH_DIR}/plot_ratios.py -i "${SIM_DIR}/sim_record.csv"
'''

        with open(curr_file, 'w') as f:
            f.write(script)

        i+=1

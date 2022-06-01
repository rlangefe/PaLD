#!/bin/bash

PY_DIR=/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/python

cd $PY_DIR
bash combine_output.sh

cd ../matlab

#bash combine_output.sh
bash combine_strong.sh

cd $PY_DIR
python plot_speedup.py
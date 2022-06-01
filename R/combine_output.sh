#!/bin/bash

OUTPUTFILE=full_run_test.csv

HEADER="\"indexVal\" \"mean1\" \"std1\" \"dist1\" \"n1\" \"dim1\" \"mean2\" \"std2\" \"dist2\" \"n2\" \"dim2\" \"time_pald_R\" \"bound_pald_R\""
echo $HEADER > $OUTPUTFILE

for NAME in $(find sim_output_files | grep ".csv")
do
cat $NAME >> $OUTPUTFILE
done
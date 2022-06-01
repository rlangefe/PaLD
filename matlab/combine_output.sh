#!/bin/bash

OUTPUTFILE=full_run_test.csv
HEADER="\"indexVal\",\"mean1\",\"std1\",\"dist1\",\"n1\",\"dim1\",\"mean2\",\"std2\",\"dist2\",\"n2\",\"dim2\",\"matlab_time\",\"matlab_bound\""
echo $HEADER > $OUTPUTFILE

for NAME in $(find runs | grep ".csv")
do
cat $NAME >> $OUTPUTFILE
done
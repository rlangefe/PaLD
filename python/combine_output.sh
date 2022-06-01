#!/bin/bash

OUTPUTFILE=full_run_test_cpu.csv
HEADER="indexVal,mean1,std1,n1,dim1,dist1,mean2,std2,n2,dim2,dist2,time_pald_numpy,bound_pald_numpy"
echo $HEADER > $OUTPUTFILE

for NAME in $(find runs | grep ".csv")
do
cat $NAME >> $OUTPUTFILE
done
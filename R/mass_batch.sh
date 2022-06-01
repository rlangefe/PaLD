#!/bin/bash

##########################################
# Need to get to 161 at max (bc 162,000) #
##########################################


# Last done was 0 at 4/1/22 1:58 pm
# Last done was 1 to 2 at 4/1/22 2:10 pm
# Last done was 3 to 4 at 4/1/22 2:10 pm
# Last done was 5 to 6 at 4/1/22 2:10 pm
# Last done was 7 to 8 at 4/1/22 2:11 pm
# Last done was 9 to 10 at 4/1/22 2:34 pm
# Last done was 11 at 4/1/22 2:34 pm
# Last done was 12 at 4/1/22 2:51 pm
# Last done was 13 to 14 at 4/1/22 5:37 pm
# Last done was 15 to 16 at 4/1/22 5:37 pm
# Last done was 17 to 18 at 4/1/22 5:37 pm
# Last done was 19 to 20 at 4/1/22 5:37 pm
# Last done was 21 to 29 at 4/4/22 12:19 pm
# Last done was 30 to 38 at 4/5/22 2:02 pm
# Last done was 39 to 47 at 4/5/22 2:02 pm
# Last done was 48 to 56 at 4/6/22 9:08 am
# Last done was 57 to 62 at 4/6/22 12:30 pm
# Last done was 63 to 65 at 4/6/22 12:31 pm
# Last done was 66 to 74 at 4/6/22 5:15 pm
# Last done was 75 to 83 at 4/7/22 7:08 am
# Last done was 84 to 92 at 4/7/22 12:29 pm
# Last done was 93 to 101 at 4/8/22 9:41 am
# Last done was 102 to 110 at 4/8/22 6:26 pm
# Last done was 111 to 119 at 4/9/22 7:46 am
# Last done was 120 to 128 at 4/9/22 2:08 pm
# Last done was 129 to 137 at 4/9/22 8:54 pm
# Last done was 138 to 146 at 4/10/22 8:00 am
# Last done was 147 to 155 at 4/10/22 11:00 am
# Last done was 156 to 162 at 4/10/22 3:14 pm
for i in $(seq 156 162)
do
    echo "Running ${i}"
    export CURR_ITER=$i
    sbatch run_R_sim.sh
    echo
done

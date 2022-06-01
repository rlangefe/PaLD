import os
import subprocess

import argparse

import pandas as pd

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-p","--positive",dest="pos",help="Positive input file", default='pos_test.txt')
    parse.add_argument("-n","--negative",dest="neg",help="Negative input file", default='neg_test.txt')
    parse.add_argument("-s","--sample",dest="sample",help="Percentage of points to sample from neg", default=100, type=int)
    parse.add_argument("-r","--radius",dest="r",help="Radius from point to check", default=2000, type=float)
    args = parse.parse_args()

    plen = len(pd.read_csv(str(args.pos), header=None))
    nlen = len(pd.read_csv(str(args.neg), header=None))

    with open('temp.sh', 'w') as f:
        f.write('./main.x ' + str(args.sample) + ' ' + str(plen) + ' ' + str(nlen) + ' ' + str(args.pos) + ' ' + str(args.neg) + ' ' + str(args.r) + '\n')

    #os.system('./main.x ' + str(args.sample) + ' ' + str(plen) + ' ' + str(nlen) + ' ' + str(args.pos) + ' ' + str(args.neg) + ' ' + str(args.r))

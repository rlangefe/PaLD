import pandas as pd
import numpy as np
import time
import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-n","--number",dest="number", help="Number of samples", default=1000, type=int)
    parse.add_argument("-d","--dim",dest="dim", help="Number of dimensions", default=1, type=int)
    parse.add_argument("-m","--mean",dest="mean", help="Mean of distribution", default=0, type=float)
    parse.add_argument("-s","--std",dest="std", help="Standard deviation of distribution", default=1, type=float)
    parse.add_argument("-r","--random",dest="random", help="Random seed to use", default=None, type=int)
    parse.add_argument("-o","--output",dest="output", help="Output file", default='../data/generated.csv', type=str)
    parse.add_argument("-x","--dist",dest="dist", help="Distribution name", default='normal', type=str)
    args = parse.parse_args()

    mean = args.mean
    std = args.std
    n = args.number
    dim = args.dim

    if args.random is not None:
        print('Using random seed', args.random)
        np.random.seed(args.random)

    data = np.zeros((n, dim))

    for d in range(dim):
        if args.dist == 'normal':
            data[:,d] = np.random.normal(mean, std, n)
        elif args.dist == 'exponential':
            data[:,d] = np.random.exponential(mean, n)
        elif args.dist == 'chisquare':
            data[:,d] = np.random.chisquare(mean, n)

    pd.DataFrame(data).to_csv(args.output, header=False, index=False)


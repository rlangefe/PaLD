import os   
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
    parse.add_argument("-o","--output",dest="output", help="Output file", default='../results/sim_record.csv', type=str)
    parse.add_argument("-i","--input",dest="input", help="Input file", default='../results/results.csv', type=str)
    args = parse.parse_args()

    df = pd.read_csv(args.input)

    isolated_count = len(set(range(args.number)) - (set(df['rows']).union(set(df['cols']))))

    if not os.path.isfile(args.output):
        data_dict = {
            'Number' : [args.number],
            'Dimensions' : [args.dim],
            'Mean' : [args.mean],
            'Std' : [args.std],
            'Isolated' : [isolated_count]
        }
        
        record = pd.DataFrame.from_dict(data_dict)
        record.to_csv(args.output, index=False)
    else:
        data_dict = {
            'Number' : args.number,
            'Dimensions' : args.dim,
            'Mean' : args.mean,
            'Std' : args.std,
            'Isolated' : isolated_count
        }

        record = pd.read_csv(args.output)
        record = record.append(data_dict, ignore_index=True)
        record.to_csv(args.output, index=False)

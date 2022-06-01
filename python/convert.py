import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-r","--rows",dest="rows", help="Rows file", default='../cuda/pald/rows.dat', type=str)
    parse.add_argument("-c","--cols",dest="cols", help="Columns file", default='../cuda/pald/cols.dat', type=str)
    parse.add_argument("-v","--values",dest="values", help="Values file", default='../cuda/pald/values.dat', type=str)
    parse.add_argument("-o","--output",dest="output", help="Output file", default='../results/results.csv', type=str)
    args = parse.parse_args()

    rows = np.frombuffer(np.fromfile(args.rows, dtype='i4'), dtype='i4').astype(np.int32)
    cols = np.frombuffer(np.fromfile(args.cols, dtype='i4'), dtype='i4').astype(np.int32)
    vals = np.frombuffer(np.fromfile(args.values, dtype='f4'), dtype='f4').astype(np.float32)

    val_dict = {'rows' : rows, 'cols' : cols, 'vals' : vals}

    df = pd.DataFrame.from_dict(val_dict)
    
    #df.to_csv('../../R/results.csv', index=False)
    df.to_csv(args.output, index=False)

import pandas as pd
import numpy as np
import time
import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-f1","--file1",dest="file1", help="File 1", default='g1.csv', type=str)
    parse.add_argument("-f2","--file2",dest="file2", help="File 2", default='g2.csv', type=str)
    parse.add_argument("-o","--output",dest="output", help="Output file", default='data.csv', type=str)
    parse.add_argument("-g","--groups",dest="groups", help="Groups file", default='groups.csv', type=str)
    args = parse.parse_args()

    g1 = pd.read_csv(args.file1, header=None)
    g2 = pd.read_csv(args.file2, header=None)

    groups = [1]*len(g1) + [2]*len(g2)

    df = pd.concat([g1,g2],ignore_index=True)

    df.to_csv(args.output, header=False, index=False)
    
    pd.DataFrame(groups).to_csv(args.groups, header=False, index=False)

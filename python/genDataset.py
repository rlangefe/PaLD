import pandas as pd
import numpy as np
import time
import argparse

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-i","--index",dest="index", help="Index of run", default=0, type=int)
    parse.add_argument("-f","--file",dest="file", help="File to pull info from", default='experiments.csv', type=str)
    args = parse.parse_args()

    info_df = pd.read_csv(args.file)

    row_config = info_df.iloc[args.index]

    print(row_config)

    for i in [j+1 for j in range(2)]:
        mean = row_config['mean' + str(i)]
        std = row_config['std' + str(i)]
        n = row_config['n' + str(i)]
        n_test = row_config['n_test' + str(i)]
        dim = row_config['dim' + str(i)]
        dist = row_config['dist' + str(i)]

        data = np.zeros((n, dim))
        
        for d in range(dim):
            if dist == 'normal':
                data[:,d] = np.random.normal(mean, std, n)
            elif dist == 'exponential':
                data[:,d] = np.random.exponential(mean, n)
            elif dist == 'chisquare':
                data[:,d] = np.random.chisquare(mean, n)

        pd.DataFrame(data).to_csv('data_g' + str(i) + '.csv', header=False, index=False)

        test_data = np.zeros((n_test, dim))
        for d in range(dim):
            if dist == 'normal':
                data[:,d] = np.random.normal(mean, std, n)
            elif dist == 'exponential':
                data[:,d] = np.random.exponential(mean, n)
            elif dist == 'chisquare':
                data[:,d] = np.random.chisquare(mean, n)

        pd.DataFrame(data).to_csv('test_g' + str(i) + '.csv', header=False, index=False)
    



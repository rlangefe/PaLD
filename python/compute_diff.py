import os
import argparse

import numpy as np
import pandas as pd

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-i","--input",dest="input", help="Input file", type=str)
    parse.add_argument("-t","--true",dest="true", help="True results file", default='', type=str)
    parse.add_argument("-r","--ru",dest="ru", help="RU file", type=str)
    parse.add_argument("-c","--c",dest="c", help="True RU file", type=str)
    args = parse.parse_args()

    results_df = pd.read_csv(args.input)
    true_df = pd.read_csv(args.true)

    with open(args.ru, 'r') as f:
        results_ru=float(f.read()[4:])

    with open(args.c, 'r') as f:
        true_ru=float(f.read()[4:])

    print("Result RU:", results_ru)

    print("True RU:", true_ru)

    print('RU Difference:', results_ru-true_ru)

    results_ties = [set(i) for i in zip(results_df['rows'], results_df['cols'])]
    true_ties = [set(i) for i in zip(true_df['rows'], true_df['cols'])]

    extra_ties = np.setdiff1d(results_ties, true_ties)#[i for i in results_ties if not (i in true_ties)]
    missing_ties = np.setdiff1d(true_ties, results_ties)#[i for i in true_ties if not (i in results_ties)]

    print("Number of Extra Elements:", len(extra_ties))
    print("Number of Missing Elements:", len(missing_ties))

    #print('Mean Absolute Weight Difference:', np.sum(np.abs(results_df['vals'].values - true_df['vals'].values))/len(true_df))

    #print('Columns in order:', not ((results_df['rows'] != true_df['rows']) | (results_df['cols'] != true_df['cols'])).any())

    
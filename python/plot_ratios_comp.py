import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-i","--input",dest="input", help="Input file", default='../results/sim_record.csv')
    parse.add_argument("-c","--compare",dest="compare", help="Compare input file", default='../results/sim_record-r.csv')
    args = parse.parse_args()

    df = pd.read_csv(args.input)

    df['Source'] = ['GPU']*len(df)

    df_comp = pd.read_csv(args.compare)

    df_comp['Source'] = ['R']*len(df_comp)

    df = df.append(df_comp, ignore_index=True)

    #df = df[df['Number']<=10000]

    ax = sns.lineplot(x="Number", y="Isolated", hue='Source', data=df)
    ax.set_title('Number vs. Isolated Count')

    plt.savefig('num_vs_isolated.png')

    plt.close()

    ax = sns.lineplot(x=df['Number'], y=df['Isolated']/df['Number'], hue=df['Source'])
    ax.set_xlabel('Number')
    ax.set_ylabel('Isolated Ratio')
    ax.set_title('Number vs. Isolated Ratio')

    plt.savefig('num_vs_ratio.png')

    plt.close()

    ax = sns.lineplot(x=df['Number'], y=df['Mean'], hue=df['Source'])
    ax.set_xlabel('Number')
    ax.set_ylabel('Mean')
    ax.set_title('Number vs. Mean')

    plt.savefig('num_vs_mean.png')
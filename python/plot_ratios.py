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
    args = parse.parse_args()

    df = pd.read_csv(args.input)

    df = df[df['Number']<=10000]

    ax = sns.regplot(x="Number", y="Isolated", data=df)
    ax.set_title('Number vs. Isolated Count')

    plt.savefig('num_vs_isolated.png')

    plt.close()

    ax = sns.regplot(x=df['Number'], y=df['Isolated']/df['Number'])
    ax.set_xlabel('Number')
    ax.set_ylabel('Isolated Ratio')
    ax.set_title('Number vs. Isolated Ratio')

    plt.savefig('num_vs_ratio.png')
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
    parse.add_argument("-p","--positive",dest="pos", help="Positive input file", default='pos_hotlist.txt')
    parse.add_argument("-n","--negative",dest="neg", help="Negative input file", default='neg_hotlist.txt')
    parse.add_argument("-o","--output",dest="output", help="Output file name", default='plot.png', type=str)
    args = parse.parse_args()

    pos = pd.read_csv(str(args.pos))
    neg = pd.read_csv(str(args.neg))

    cmap = matplotlib.cm.get_cmap('YlOrRd')

    coloraxlist1 = (pos['LD']-pos['LD'].min())/(pos['LD'].max()-pos['LD'].min())*17+3
    #coloraxlist1 = [cmap(i) for i in coloraxlist1.values]

    # sns.scatterplot(
    #     x=np.hstack((pos['X'].values, neg['X'].values)),
    #     y=np.hstack((pos['Y'].values, neg['Y'].values)),
    #     hue=list(coloraxlist1.values)+['b']*len(neg)
    #     )

    #plt.scatter(pos['X'].values, pos['Y'].values, c=coloraxlist1, marker=',', s=2)
    #plt.scatter(pos[pos['Hot'] == 1]['X'].values, pos[pos['Hot'] == 1]['Y'].values, cmap=cmap, vmin=pos['LD'].min(), vmax=pos['LD'].max(), marker=',', s=1)
    #plt.scatter(pos[pos['Hot'] == 0]['X'].values, pos[pos['Hot'] == 0]['Y'].values, cmap=cmap, vmin=pos['LD'].min(), vmax=pos['LD'].max(), marker='.', s=1)
    plt.scatter(neg['X'].values, neg['Y'].values, c='blue', marker='.', s=1)
    plt.scatter(pos[pos['Hot'] == 1]['X'].values, pos[pos['Hot'] == 1]['Y'].values, cmap=cmap, c=pos[pos['Hot'] == 1]['LD'], marker=',', s=1)
    plt.scatter(pos[pos['Hot'] == 0]['X'].values, pos[pos['Hot'] == 0]['Y'].values, cmap=cmap, c=pos[pos['Hot'] == 0]['LD'], marker='.', s=1)

    plt.savefig(str(args.output))

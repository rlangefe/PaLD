import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    data = pd.read_csv('results.csv')
    meta_data = pd.read_csv('info.csv').iloc[range(len(data))]

    fig = plt.figure(figsize=(12,12))

    for predictor, i in zip(data.columns, range(1,10)):
        ax = fig.add_subplot(3,3,i,projection='3d')
        sc = ax.scatter(xs=meta_data['std1'], ys=meta_data['std2'], zs=meta_data['mean2']-meta_data['mean1'], marker='o', c=data[predictor], alpha=1)
        ax.set_xlabel(r'$\sigma_1$')
        ax.set_ylabel(r'$\sigma_2$')
        ax.set_zlabel(r'$|\mu_1 - \mu_2|$')

        # legend
        ax.legend(*sc.legend_elements())
        #ax.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

        ax.set_title(predictor)

    fig.tight_layout()
    plt.savefig('graphs.png')

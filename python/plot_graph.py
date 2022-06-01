import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if __name__ == '__main__':
    edges = pd.read_csv('results.csv')
    points = pd.read_csv('dataset_testing.csv', header=False)
    clusters = pd.read_csv('clusters.csv', header=False)

    x = StandardScaler().fit_transform(points.values)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    plt.scatter(x[:,0], x[:,1], c=clusters['Cluster'])

    for idx, val in edges.iterrows():
        x = [principalComponents[val[0],0], principalComponents[val[1],0]]
        y = [principalComponents[val[0],1], principalComponents[val[1],1]]
        plt.plot(x, y, c=val[3], cmap='Greys', alpha=0.2)

    plt.savefig('graph.png')


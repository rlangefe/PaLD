import numpy as np
import cupy as cp
import pandas as pd

import pprint
import os
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.colors import ListedColormap
import seaborn as sns

run_type = 'gpu'

if run_type == 'cpu':
    print('CPU Run')
    from pald_numpy import pald
    #from pald_numpy_parallel import pald
elif run_type == 'gpu':
    print('GPU Run')
    #from pald_cupy_vectorized import pald
    #from pald_cupy import pald
    from pald_cupy_loop import pald
    #from pald_cupy_multigpu import pald
    #from pald_cupy_multistream import pald
    #from pald_cupy_custom import pald
    #from pald_cupy_unrolled import pald

from scipy.spatial.distance import pdist, squareform

dummy_kernel = cp.RawKernel(r'''
extern "C" __global__
void dummy_kernel() {
    
}
''','dummy_kernel')

if __name__ == '__main__':
    print('Reading data')
    #df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/sonar/data.csv', header=None)
    #df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/lang.csv', header=None)
    #df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/7500/7500pts.csv', header=None)
    #df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/TCGA-PANCAN-HiSeq-801x20531/data.csv')
    #df = df[df.columns[2:]]
    df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/python/pballs_big.csv')
    df = df[['V1', 'V2']]
    print('Data length:', len(df))

    print('Calculating distances')
    dist_mat = squareform(pdist(df))
    #dist_mat = df.values
    print('Running PaLD')
    
    if run_type == 'cpu':
        start_time = time.time()
        cohesions = pald(dist_mat)
        end_time = time.time()
    elif run_type == 'gpu':
        with cp.cuda.Device(0) as dev:
            dummy_kernel((10,), (10,), ())
            start_time = time.time()
            cohesions = pald(dist_mat)
            end_time = time.time()
    
    if run_type == 'gpu':
        cohesions = cohesions.get()

    #pprint.pprint(cohesions)
    print("--- %s seconds ---" % (end_time - start_time))
    print('Bound:', cohesions.trace()/(dist_mat.shape[0]*2))
    #pd.DataFrame(cohesions).to_csv('sonar_test.csv', index=False, header=False)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
    sns.scatterplot('V1', 'V2', hue=kmeans.predict(df), data=df)
    plt.savefig('kmeans_run.png')
    plt.close()

    sns.heatmap(cohesions)
    plt.savefig('cohesion_heatmap.png')

    

    

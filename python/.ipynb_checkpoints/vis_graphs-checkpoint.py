import numpy as np
import pandas as pd

#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt
#import matplotlib.colors as clr

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import cuxfilter
import cudf

print('Reading Edges')
edges = pd.read_csv('../R/results.csv')
print('Reading Points')
points = pd.read_csv('../cuda/pald/dataset_testing.csv', header=None)
#clusters = pd.read_csv('clusters.csv', header=False)

print('Normalizing')
x = StandardScaler().fit_transform(points.values)

print('Running PCA')
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

print('Forming Device Edge List')
d_edges = cudf.DataFrame({'source': edges['rows'].values, 'target': edges['cols'].values, 'weight': edges['vals'].values})

print('Forming Device Node List')
d_nodes = cudf.DataFrame({'vertex': range(len(points)), 'x': x[:,0], 'y': x[:,1]})

print('Loading Graph')
cux_df = cuxfilter.DataFrame.load_graph((d_nodes, d_edges))

print('Building Chart')
chart0 = cuxfilter.charts.datashader.graph(node_pixel_shade_type='linear')

print('Constructing Dashboard')
d = cux_df.dashboard([chart0], layout=cuxfilter.layouts.double_feature)

print('Displaying')
chart0.view()

d.app('localhost:8888')
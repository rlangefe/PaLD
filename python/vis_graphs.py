import numpy as np
import pandas as pd
import bokeh

from random import randint

#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt
#import matplotlib.colors as clr

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import cuxfilter
import cudf
import cugraph

print('Reading Edges')
edges = pd.read_csv('../rna_seq.csv')
print(str(len(edges)) + ' Edges')

print('Reading Points')
points = pd.read_csv('../data/TCGA-PANCAN-HiSeq-801x20531/data_mod.csv', header=None)
#points = pd.read_csv('../cuda/pald/dataset_testing.csv', header=None)
#points = pd.read_csv('../cuda/pald/unbalance.csv', header=None)
#points = pd.read_csv('../data/50pts.csv', header=None)
#points = pd.read_csv('../cuda/pald/data.csv', header=None)
print(str(len(points)) + ' Points')

#print('Reading Clusters')
#clusters = pd.read_csv('../cuda/pald/clusters.txt', header=None)

print('Normalizing')
x = StandardScaler().fit_transform(points.values)

print('Running PCA')
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalComponents = x
print('Forming Device Edge List')
d_edges = cudf.DataFrame({'source': edges['rows'].values, 'target': edges['cols'].values, 'weight': edges['vals'].values})

print('Calculating Clusters')
G = cugraph.Graph()
G.from_cudf_edgelist(d_edges, 'source', 'target')
G.add_nodes_from(range(len(points)))
G.to_undirected()
#G = cugraph.layout.force_atlas2(G)
df = cugraph.connected_components(G, connection="strong")

print('Assigning Clusters')
clusters = np.zeros(len(points), dtype=np.int32)
clusters[df['vertex'].to_array()]  = df['labels'].to_array()
max_val = df['labels'].to_array().max()
idx = [x for x in range(len(points)) if x not in df['vertex'].to_array()]
clusters[idx] = np.array(range(len(idx)))+max_val+1

print('Forming Device Node List')
#d_nodes = cudf.DataFrame({'vertex': range(len(points)), 'x': x[:,0], 'y': x[:,1], 'cluster': clusters.values[:,0]})
d_nodes = cudf.DataFrame({'vertex': range(len(points)), 'x': x[:,0], 'y': x[:,1], 'cluster': clusters})

print('Loading Graph')
cux_df = cuxfilter.DataFrame.load_graph((d_nodes, d_edges)
#cux_df = cugraph.layout.force_atlas2(cux_df)

colors = []

for i in range(len(np.unique(clusters))):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

print('Building Chart')
chart0 = cuxfilter.charts.datashader.graph(node_pixel_shade_type='linear', node_aggregate_fn='mean',
                                            node_color_palette=colors, node_aggregate_col='cluster', #edge_transparency=0.5, 
                                            edge_color_palette=list(reversed(bokeh.palettes.Greys9))[3:],
                                            edge_aggregate_col='weight')

print('Constructing Dashboard')
d = cux_df.dashboard([chart0], layout=cuxfilter.layouts.double_feature)

print('Displaying')
chart0.view()

d.show('http://localhost:8888')
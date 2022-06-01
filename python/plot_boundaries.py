import numpy as np
import cupy as cp
import pandas as pd

import pprint
import os
import time
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.colors import ListedColormap
from matplotlib import cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.datasets import load_digits, load_diabetes, load_breast_cancer, fetch_openml, load_iris

# ML models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

import warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning) 

run_type = 'gpu'

if run_type == 'cpu':
    from pald_numpy import pald
elif run_type == 'gpu':
    #from pald_cupy_vectorized import pald
    #from pald_cupy import pald
    #from pald_cupy_multigpu import pald
    #from pald_cupy_multistream import pald
    #from pald_cupy_custom import pald
    from pald_cupy_loop import pald
    #from pald_cupy_unrolled import pald

from scipy.spatial.distance import pdist, squareform

dummy_kernel = cp.RawKernel(r'''
extern "C" __global__
void dummy_kernel() {
    
}
''','dummy_kernel')

if __name__ == '__main__':
    #plt.rcParams['axes.grid'] = False

    with cp.cuda.Device(0) as dev:
        plt.style.use('seaborn-whitegrid')

        print('Reading data')
        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/pballs.csv')
        # df = df[['V1', 'V2']]
        # groups = np.array([0]*100 + [1]*100)

        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/python/pballs_big.csv')
        # df = df[['V1', 'V2']]
        # groups = np.array([0]*400 + [1]*400)

        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/fig4c_data.csv')
        # df = df[['V1', 'V2']]
        # groups = np.array([0]*250 + [1]*250)

        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/GazelleData.csv')
        # groups = list(df['zebra.present'].values)
        # df = df[['x','y']]

        
        #df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/sim/data.csv', header=None)
        #groups = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/sim/groups.csv', header=None).values.reshape(-1)

        # digits = load_digits() # fetch_openml("glass", return_X_y=False, data_home='/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/mnist')
        # df = pd.DataFrame(digits['data'])
        # groups = np.array(list(digits['target']))

        #########
        # Moons #
        #########
        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/moons.csv')
        # groups = np.array(df['moonsgt'])
        # df = df[['V2', 'V3']]

        # Generate Random Data
        # df, groups = make_classification(n_classes=5,
        #                                 n_features=25,
        #                                 n_samples=500,
        #                                 n_clusters_per_class=1,
        #                                 n_informative=15,
        #                                 n_redundant=5,
        #                                 flip_y=0.05,
        #                                 class_sep=3,
        #                                 hypercube=True, 
        #                                 random_state=42)
        # groups = groups
        # df = pd.DataFrame(df)

        df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/sonar/data.csv', header=None)
        groups = np.array([1 if i == 'R' else 2 for i in pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/sonar/groups.csv', header=None).values])
        #df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/lang.csv', header=None)
        #df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/7500/7500pts.csv', header=None)
        #print(df)

        print('Scaling Data')
        scaler = StandardScaler()
        scaler = scaler.fit(df)
        df = scaler.transform(df)

        print('Running PCA')
        pca = PCA(n_components=2)
        df = pca.fit_transform(df)
        # print('Running t-SNE')
        # tsne = TSNE(n_components=2, 
        #             verbose=1, 
        #             perplexity=40, 
        #             n_iter=500,
        #             random_state=42)
        # df = tsne.fit_transform(df)
        print('Data length:', len(df))

        groups = [list(np.unique(groups)).index(i) for i in groups]

        print('Plotting Original Points')
        h = np.min([df[:, 0].max() + 1 - df[:, 0].min() - 1, df[:, 1].max() + 1 - df[:, 1].min() - 1])/30

        x_min, x_max = df[:, 0].min() - 1, df[:, 0].max() + 1
        y_min, y_max = df[:, 1].min() - 1, df[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        ax = sns.scatterplot(x=df[:,0], y=df[:,1], hue=groups)

        plt.savefig(os.path.join('boundary_plots', 'pred_points.png'))
        plt.close()


        data_dict = {
                        'x' : [],
                        'y' : [],
                        'bound' : [],
                        'time' : []
                    }

        for g in np.unique(np.array(groups)):
            group_list = [name.replace('g1', str(g)) for name in ['g1_to_sum_strong',
                                                                    'g1_to_num_strong',
                                                                    'g1_from_sum_strong',
                                                                    'g1_from_num_strong',
                                                                    'g1_to_sum_total',
                                                                    'g1_from_sum_total',
                                                                    'g1_to_weighted_second',
                                                                    'g1_from_weighted_second',
                                                                    'g1_to_max',
                                                                    'g1_from_max']]

            for name in group_list:
                data_dict[name] = []

        print('Computing Distances')
        full_mat = squareform(pdist(pd.DataFrame(np.vstack((df,np.c_[xx.ravel(), yy.ravel()])))))

        if run_type == 'gpu':
            full_mat = cp.array(full_mat)

        extra_pts = np.c_[xx.ravel(), yy.ravel()]

        print('Computing PaLD Sample Weights')
        if run_type == 'cpu':
            start_time = time.time()
            cohesions = pald(full_mat[0:len(df), 0:len(df)])
            end_time = time.time()
        elif run_type == 'gpu':
            cohesions = pald(full_mat[0:len(df), 0:len(df)], block_size=32, dev=dev).get()

        weights = np.sum(cohesions>=(cohesions.trace()/(cohesions.shape[0]*2)), axis=0)

        for a in tqdm(range(len(extra_pts))):
            dist_mat = full_mat[tuple(np.meshgrid(list(range(0,len(df))) + [len(df)+a], list(range(0,len(df))) + [len(df)+a]))]
            if run_type == 'cpu':
                start_time = time.time()
                cohesions = pald(dist_mat)
                end_time = time.time()
            elif run_type == 'gpu':
                with cp.cuda.Device(0) as dev:
                    dummy_kernel((10,), (10,), ())
                    start_time = time.time()
                    cohesions = pald(dist_mat, block_size=32, dev=dev)
                    end_time = time.time()

            #pprint.pprint(cohesions)
            #print("--- %s seconds ---" % (time.time() - start_time))
            if run_type == 'gpu':
                cohesions = cohesions.get()
            
            bound = cohesions.trace()/(dist_mat.shape[0]*2)
            #print('Bound:', bound)
            
            data_dict['x'].append(extra_pts[a,0])
            data_dict['y'].append(extra_pts[a,1])
            data_dict['bound'].append(bound)

            if run_type == 'cpu':
                for g in np.unique(np.array(groups)):
                    data_dict['g1_to_sum_strong'.replace('g1', str(g))].append(np.sum(cohesions[-1,:-1][(cohesions[-1,:-1] >= bound) & (groups == g)]))

                    data_dict['g1_to_num_strong'.replace('g1', str(g))].append(np.sum((cohesions[-1,:-1] >= bound) & (groups == g)))

                    data_dict['g1_from_sum_strong'.replace('g1', str(g))].append(np.sum(cohesions[:-1,-1][(cohesions[:-1,-1] >= bound) & (groups == g)]))

                    data_dict['g1_from_num_strong'.replace('g1', str(g))].append(np.sum((cohesions[:-1,-1] >= bound) & (groups == g)))

                    data_dict['g1_to_sum_total'.replace('g1', str(g))].append(np.sum(cohesions[-1,:-1][groups == g]))

                    data_dict['g1_from_sum_total'.replace('g1', str(g))].append(np.sum(cohesions[:-1,-1][groups == g]))

                    data_dict['g1_to_weighted_second'.replace('g1', str(g))].append(np.sum(cohesions[:-1, :-1] * ((groups == g).T & (groups == g)) * cohesions[-1, :-1]))

                    data_dict['g1_from_weighted_second'.replace('g1', str(g))].append(np.sum(cohesions[:-1, :-1] * ((groups == g).T & (groups == g)) * cohesions[:-1, -1]))

                    data_dict['g1_to_max'.replace('g1', str(g))].append(np.max(cohesions[-1,:-1][groups == g]))

                    data_dict['g1_from_max'.replace('g1', str(g))].append(np.max(cohesions[:-1,-1][groups == g]))

                data_dict['time'].append(end_time-start_time)
            elif run_type == 'gpu':
                for g in np.unique(np.array(groups)):
                    data_dict['g1_to_sum_strong'.replace('g1', str(g))].append(np.sum(cohesions[-1,:-1][(cohesions[-1,:-1] >= bound) & (groups == g)]))

                    data_dict['g1_to_num_strong'.replace('g1', str(g))].append(np.sum((cohesions[-1,:-1] >= bound) & (groups == g)))

                    data_dict['g1_from_sum_strong'.replace('g1', str(g))].append(np.sum(cohesions[:-1,-1][(cohesions[:-1,-1] >= bound) & (groups == g)]))

                    data_dict['g1_from_num_strong'.replace('g1', str(g))].append(np.sum((cohesions[:-1,-1] >= bound) & (groups == g)))

                    data_dict['g1_to_sum_total'.replace('g1', str(g))].append(np.sum(cohesions[-1,:-1][groups == g]))

                    data_dict['g1_from_sum_total'.replace('g1', str(g))].append(np.sum(cohesions[:-1,-1][groups == g]))

                    data_dict['g1_to_weighted_second'.replace('g1', str(g))].append(np.sum((cohesions[:-1, :-1] >= bound) * ((groups == g).T & (groups == g)) * cohesions[-1, :-1]))

                    data_dict['g1_from_weighted_second'.replace('g1', str(g))].append(np.sum((cohesions[:-1, :-1] >= bound) * ((groups == g).T & (groups == g)) * cohesions[:-1, -1]))
                    
                    data_dict['g1_to_max'.replace('g1', str(g))].append(np.max(cohesions[-1,:-1][groups == g]))

                    data_dict['g1_from_max'.replace('g1', str(g))].append(np.max(cohesions[:-1,-1][groups == g]))

                data_dict['time'].append(end_time-start_time)
        
        pd.DataFrame().from_dict(data_dict, orient='columns').to_csv('mesh_results.csv', index=False)

        pred_pairs = ['to_sum_strong', 
                        'to_num_strong', 
                        'from_sum_strong', 
                        'from_num_strong', 
                        'to_weighted_second', 
                        'from_weighted_second', 
                        'to_sum_total', 
                        'from_sum_total',
                        'to_max',
                        'from_max']
        

        cmap_name = 'Set3'

        for name in ['from_max']:            
            Z_temp = np.array([data_dict[str(group_name) + '_' + name] for group_name in np.unique(np.array(groups))])
            Z_temp = [list(range(len(np.unique(np.array(groups)))))[idx] if count == 1 else len(np.unique(groups))+1 for idx, count in zip(np.argmax(Z_temp, axis=0), np.sum(np.max(Z_temp, keepdims=True, axis=0)==Z_temp, axis=0))]

            to_max_arr = np.array(Z_temp)

        for name in pred_pairs:
            print('Plotting ' + name)
            cm_temp = cm.get_cmap('gist_rainbow')
            cmap_light = {g : color for g, color in zip(list(range(len(np.unique(np.array(groups))))), cm.get_cmap(cmap_name, len(np.unique(np.array(groups)))).colors)}
            cmap_bold = {g : color for g, color in zip(list(range(len(np.unique(np.array(groups))))), cm.get_cmap(cmap_name, len(np.unique(np.array(groups)))).colors)}
            cmap_light[len(np.unique(groups))+1] = 'white'
            
            Z_temp = np.array([data_dict[str(group_name) + '_' + name] for group_name in np.unique(np.array(groups))])
            Z_temp = [list(range(len(np.unique(np.array(groups)))))[idx] if count == 1 else max_val for idx, count, max_val in zip(np.argmax(Z_temp, axis=0), np.sum(np.max(Z_temp, keepdims=True, axis=0)==Z_temp, axis=0), to_max_arr)]

            Z = np.array(Z_temp)
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            
            # plt.imshow(Z, 
            #         alpha=0.6,
            #         extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            #         interpolation='nearest',
            #         cmap=ListedColormap([cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups))))))]),
            #         origin='lower')
            plt.pcolormesh(xx, yy, Z, alpha=0.6, shading='nearest',
                            cmap=ListedColormap([cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups))))))]))

            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            
            sns.scatterplot(x=df[:,0], y=df[:,1], hue=groups, style=groups, palette={np.unique(groups)[i] : cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups)))))) if i != len(np.unique(groups))+1}, s=20, edgecolor="black")
            plt.title('Decision Boundary for ' + name.replace('_', ' ').title())
            plt.savefig(os.path.join('boundary_plots', name + '.png'))
            plt.close()

        for name in pred_pairs:
            print('Plotting ' + name)
            cm_temp = cm.get_cmap('gist_rainbow')
            cmap_light = {g : color for g, color in zip(list(range(len(np.unique(np.array(groups))))), cm.get_cmap(cmap_name, len(np.unique(np.array(groups)))).colors)}
            cmap_bold = {g : color for g, color in zip(list(range(len(np.unique(np.array(groups))))), cm.get_cmap(cmap_name, len(np.unique(np.array(groups)))).colors)}
            cmap_light[len(np.unique(groups))+1] = 'white'
            
            Z_temp = np.array([data_dict[str(group_name) + '_' + name] for group_name in np.unique(np.array(groups))])
            Z_temp = [list(range(len(np.unique(np.array(groups)))))[idx] if count == 1 else len(np.unique(groups))+1 for idx, count, max_val in zip(np.argmax(Z_temp, axis=0), np.sum(np.max(Z_temp, keepdims=True, axis=0)==Z_temp, axis=0), to_max_arr)]

            Z = np.array(Z_temp)
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            
            # plt.imshow(Z, 
            #         alpha=0.6,
            #         extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            #         interpolation='nearest',
            #         cmap=ListedColormap([cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups))))))]),
            #         origin='lower')
            plt.pcolormesh(xx, yy, Z, alpha=0.6, shading='nearest',
                            cmap=ListedColormap([cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups))))))]))

            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            
            sns.scatterplot(x=df[:,0], y=df[:,1], hue=groups, style=groups, palette={np.unique(groups)[i] : cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups)))))) if i != len(np.unique(groups))+1}, s=20, edgecolor="black")
            plt.title('Decision Boundary for ' + name.replace('_', ' ').title())
            plt.savefig(os.path.join('boundary_plots', name + '_alone.png'))
            plt.close()

        list_of_models = ['knn_3',
                            'knn_5',
                            'log_reg',
                            'svc',
                            'decision_tree',
                            'log_reg_weighted',
                            'svc_weighted',
                            'decision_tree_weighted']

        # Standard ML Models
        for name in list_of_models:
            print('Plotting ' + name)
            cm_temp = cm.get_cmap('gist_rainbow')
            cmap_light = {g : color for g, color in zip(list(range(len(np.unique(np.array(groups))))), cm.get_cmap(cmap_name, len(np.unique(np.array(groups)))).colors)}
            cmap_bold = {g : color for g, color in zip(list(range(len(np.unique(np.array(groups))))), cm.get_cmap(cmap_name, len(np.unique(np.array(groups)))).colors)}
            cmap_light[len(np.unique(groups))+1] = 'white'

            if name == 'knn_3':
                clf = KNeighborsClassifier(n_neighbors=3).fit(df,groups)

            elif name == 'knn_5':
                clf = KNeighborsClassifier(n_neighbors=5).fit(df,groups)

            elif name == 'log_reg':
                clf = LogisticRegression(multi_class='ovr').fit(df,groups)
            
            elif name == 'svc':
                clf = SVC(gamma='auto').fit(df,groups)
            
            elif name == 'decision_tree':
                clf = DecisionTreeClassifier(random_state=0).fit(df,groups)

            elif name == 'log_reg_weighted':
                clf = LogisticRegression(multi_class='ovr').fit(df,groups, sample_weight=weights)
            
            elif name == 'svc_weighted':
                clf = SVC(gamma='auto').fit(df,groups, sample_weight=weights)
            
            elif name == 'decision_tree_weighted':
                clf = DecisionTreeClassifier(random_state=0).fit(df,groups, sample_weight=weights)
            

            Z_temp = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = np.array(Z_temp)
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            
            # plt.imshow(Z, 
            #         alpha=0.6,
            #         extent=[xx.min(), xx.max(), yy.min(), yy.max()],
            #         interpolation='nearest',
            #         cmap=ListedColormap([cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups))))))]),
            #         origin='lower')
            plt.pcolormesh(xx, yy, Z, alpha=0.6, shading='nearest',
                            cmap=ListedColormap([cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups))))))]))

            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())

            sns.scatterplot(x=df[:,0], y=df[:,1], hue=groups, style=groups, palette={np.unique(groups)[i] : cmap_light[i] for i in list(np.unique(list(np.unique(Z.ravel())) + list(range(len(np.unique(groups)))))) if i != len(np.unique(groups))+1}, s=20, edgecolor="black")
            plt.title('Decision Boundary for ' + name.replace('_', ' ').title().replace('Svc', 'SVC').replace('Knn', 'KNN'))
            plt.savefig(os.path.join('boundary_plots', name + '.png'))
            plt.close()
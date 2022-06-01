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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

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
    from pald_cupy_loop import pald

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
        #########
        # Moons #
        #########
        df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/moons.csv')
        groups = np.array(df['moonsgt'])
        df = df[['V2', 'V3']]


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

        #digits = fetch_openml("abalone", return_X_y=False, data_home='/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/mnist')
        #digits = fetch_openml("mfeat-fourier", return_X_y=False, data_home='/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/mnist')
        # digits = fetch_openml("vowel", return_X_y=False, data_home='/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/mnist')
        # df = pd.DataFrame(digits['data']).drop(columns=['Sex', 'Speaker_Number'])
        # df = pd.DataFrame(df, dtype=np.float32)
        # digits = load_iris() # fetch_openml("glass", return_X_y=False, data_home='/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/mnist')
        
        # df = pd.DataFrame(digits['data'])
        # # #df['Sex'] = df['Sex'] == 'M'
        # df = df.astype(np.float32)
        # groups = np.array(list(digits['target']))

        # Generate Random Data
        # df, groups = make_classification(n_classes=2,
        #                                 n_features=25,
        #                                 n_samples=500,
        #                                 n_clusters_per_class=2,
        #                                 n_informative=3,
        #                                 n_redundant=5,
        #                                 flip_y=0.05,
        #                                 class_sep=3,
        #                                 hypercube=True, 
        #                                 random_state=42)
        # groups = groups
        # df = pd.DataFrame(df)

        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/tissuesGeneExpression/data/data.csv')
        # groups = list(df['tissue'])
        # df = df.drop(columns=['tissue'])

        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/sonar/data.csv', header=None)
        # groups = np.array([1 if i == 'R' else 2 for i in pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/sonar/groups.csv', header=None).values])
        
        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/lang.csv', header=None)
        # df = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/7500/7500pts.csv', header=None)
        
        print('Data length:', len(df))

        # print('Scaling Data')
        # scaler = StandardScaler()
        # scaler = scaler.fit(df)
        # df = scaler.transform(df)
        df = df.values

        groups = [list(np.unique(groups)).index(i) for i in groups]

        X_train, X_test, y_train, y_test = train_test_split(df, groups, test_size=0.8, random_state=42)


        data_dict = {
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
        full_mat = squareform(pdist(pd.DataFrame(np.vstack((X_train, X_test)))))

        if run_type == 'gpu':
            full_mat = cp.array(full_mat)

        extra_pts = X_test

        print('Computing PaLD Sample Weights')
        if run_type == 'cpu':
            start_time = time.time()
            cohesions = pald(full_mat[0:len(X_train), 0:len(X_train)])
            end_time = time.time()
        elif run_type == 'gpu':
            cohesions = pald(full_mat[0:len(X_train), 0:len(X_train)], block_size=32, dev=dev).get()

        weights = np.sum(cohesions>=(cohesions.trace()/(cohesions.shape[0]*2)), axis=1)
        #weights = np.sum(cohesions, axis=0)

        for a in tqdm(range(len(extra_pts))):
            dist_mat = full_mat[tuple(np.meshgrid(list(range(0,len(X_train))) + [len(X_train)+a], list(range(0,len(X_train))) + [len(X_train)+a]))]
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
            
            data_dict['bound'].append(bound)

            if run_type == 'cpu':
                for g in np.unique(np.array(groups)):
                    data_dict['g1_to_sum_strong'.replace('g1', str(g))].append(np.sum(cohesions[-1,:-1][(cohesions[-1,:-1] >= bound) & (y_train == g)]))

                    data_dict['g1_to_num_strong'.replace('g1', str(g))].append(np.sum((cohesions[-1,:-1] >= bound) & (y_train == g)))

                    data_dict['g1_from_sum_strong'.replace('g1', str(g))].append(np.sum(cohesions[:-1,-1][(cohesions[:-1,-1] >= bound) & (y_train == g)]))

                    data_dict['g1_from_num_strong'.replace('g1', str(g))].append(np.sum((cohesions[:-1,-1] >= bound) & (y_train == g)))

                    data_dict['g1_to_sum_total'.replace('g1', str(g))].append(np.sum(cohesions[-1,:-1]*(y_train == g)))

                    data_dict['g1_from_sum_total'.replace('g1', str(g))].append(np.sum(cohesions[:-1,-1]*(y_train == g)))

                    data_dict['g1_to_weighted_second'.replace('g1', str(g))].append(np.sum(cohesions[:-1, :-1] * ((y_train == g).T & (y_train == g)) * cohesions[-1, :-1]))

                    data_dict['g1_from_weighted_second'.replace('g1', str(g))].append(np.sum(cohesions[:-1, :-1] * ((y_train == g).T & (y_train == g)) * cohesions[:-1, -1]))

                    data_dict['g1_to_max'.replace('g1', str(g))].append(np.max(cohesions[-1,:-1]*(y_train == g)))

                    data_dict['g1_from_max'.replace('g1', str(g))].append(np.max(cohesions[:-1,-1]*(y_train == g)))

                data_dict['time'].append(end_time-start_time)
            elif run_type == 'gpu':
                for g in np.unique(np.array(groups)):
                    data_dict['g1_to_sum_strong'.replace('g1', str(g))].append(np.sum(cohesions[-1,:-1][(cohesions[-1,:-1] >= bound) & (y_train == g)]))

                    data_dict['g1_to_num_strong'.replace('g1', str(g))].append(np.sum((cohesions[-1,:-1] >= bound) & (y_train == g)))

                    data_dict['g1_from_sum_strong'.replace('g1', str(g))].append(np.sum(cohesions[:-1,-1][(cohesions[:-1,-1] >= bound) & (y_train == g)]))

                    data_dict['g1_from_num_strong'.replace('g1', str(g))].append(np.sum((cohesions[:-1,-1] >= bound) & (y_train == g)))

                    data_dict['g1_to_sum_total'.replace('g1', str(g))].append(np.sum(cohesions[-1,:-1]*(y_train == g)))

                    data_dict['g1_from_sum_total'.replace('g1', str(g))].append(np.sum(cohesions[:-1,-1]*(y_train == g)))

                    data_dict['g1_to_weighted_second'.replace('g1', str(g))].append(np.sum((cohesions[:-1, :-1] >= bound) * ((y_train == g).T & (y_train == g)) * cohesions[-1, :-1]))

                    data_dict['g1_from_weighted_second'.replace('g1', str(g))].append(np.sum((cohesions[:-1, :-1] >= bound) * ((y_train == g).T & (y_train == g)) * cohesions[:-1, -1]))
                    
                    data_dict['g1_to_max'.replace('g1', str(g))].append(np.max(cohesions[-1,:-1]*(y_train == g)))

                    data_dict['g1_from_max'.replace('g1', str(g))].append(np.max(cohesions[:-1,-1]*(y_train == g)))

                data_dict['time'].append(end_time-start_time)
        
        #pd.DataFrame().from_dict(data_dict, orient='columns').to_csv('mesh_results.csv', index=False)

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
        

        results_dict = {}

        for name in ['from_max']:            
            Z_temp = np.array([data_dict[str(group_name) + '_' + name] for group_name in np.unique(np.array(y_train))])
            Z_temp = [list(range(len(np.unique(np.array(y_train)))))[idx] if count == 1 else len(np.unique(y_train))+1 for idx, count in zip(np.argmax(Z_temp, axis=0), np.sum(np.max(Z_temp, keepdims=True, axis=0)==Z_temp, axis=0))]

            to_max_arr = np.array(Z_temp)

        for name in pred_pairs:            
            Z_temp = np.array([data_dict[str(group_name) + '_' + name] for group_name in np.unique(np.array(y_train))])
            #Z_temp = [list(range(len(np.unique(np.array(y_train)))))[idx] if count == 1 else len(np.unique(y_train))+1 for idx, count in zip(np.argmax(Z_temp, axis=0), np.sum(np.max(Z_temp, keepdims=True, axis=0)==Z_temp, axis=0))]
            Z_temp = [list(range(len(np.unique(np.array(y_train)))))[idx] if count == 1 else max_val for idx, count, max_val in zip(np.argmax(Z_temp, axis=0), np.sum(np.max(Z_temp, keepdims=True, axis=0)==Z_temp, axis=0), to_max_arr)]

            Z = np.array(Z_temp)

            results_dict[name.replace('_', ' ').title()] = [accuracy_score(y_test, Z, normalize=True), precision_score(y_test, Z, average='macro', zero_division=0), recall_score(y_test, Z, average='macro', zero_division=0), f1_score(y_test, Z, average='macro', zero_division=0), balanced_accuracy_score(y_test, Z)]


        list_of_models = ['knn_3',
                            'knn_5',
                            'log_reg',
                            'log_reg_weighted',
                            'svc',
                            'svc_weighted',
                            'decision_tree',
                            'decision_tree_weighted']

        # Standard ML Models
        for name in list_of_models:
            if name == 'knn_3':
                clf = KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)

            elif name == 'knn_5':
                clf = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)

            elif name == 'log_reg':
                clf = LogisticRegression(multi_class='ovr').fit(X_train,y_train)
            
            elif name == 'svc':
                clf = SVC(gamma='auto').fit(X_train,y_train)
            
            elif name == 'decision_tree':
                clf = DecisionTreeClassifier(random_state=0).fit(X_train,y_train)

            elif name == 'log_reg_weighted':
                clf = LogisticRegression(multi_class='ovr').fit(X_train,y_train, sample_weight=weights)
            
            elif name == 'svc_weighted':
                clf = SVC(gamma='auto').fit(X_train,y_train, sample_weight=weights)
            
            elif name == 'decision_tree_weighted':
                clf = DecisionTreeClassifier(random_state=0).fit(X_train,y_train, sample_weight=weights)

            results_dict[name.replace('_', ' ').title().replace('Svc', 'SVC').replace('Knn', 'KNN')] = [clf.score(X_test, y_test), precision_score(y_test, clf.predict(X_test), average='macro', zero_division=0), recall_score(y_test, clf.predict(X_test), average='macro', zero_division=0), f1_score(y_test, clf.predict(X_test), average='macro', zero_division=0), balanced_accuracy_score(y_test, clf.predict(X_test))]

        results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Bal Acc'])

        print(results_df)

        sns.heatmap(results_df, annot=True)
        plt.tight_layout()
        plt.savefig('scores.png')

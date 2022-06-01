import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import softmax
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import argparse

def mean_cohesion(cohesions, groups, n):
    group_list = np.unique(groups)
    results = np.zeros((n,len(group_list)))
    group_mapping = {}
    
    for i, g in zip(range(len(group_list)),group_list):
        group_mapping[i] = g
        span_set = (groups==g)[:,0]
        
        if span_set.any():
            valid_subset = cohesions[np.isin(cohesions[:,1],np.argwhere(span_set))]
            for j in range(n):
                j_subset = valid_subset[valid_subset[:,0] == j,2]
                if j_subset.any():
                    results[j,i] = np.mean(j_subset)

    results = softmax(results, axis=1)

    results = np.argmax(results, axis=1)

    return [group_mapping[i] for i in results]


def total_cohesion(cohesions, groups, n):
    group_list = np.unique(groups)
    results = np.zeros((n,len(group_list)))
    group_mapping = {}
    
    for i, g in zip(range(len(group_list)),group_list):
        group_mapping[i] = g
        span_set = (groups==g)[:,0]
        
        if span_set.any():
            valid_subset = cohesions[np.isin(cohesions[:,1],np.argwhere(span_set))]
            for j in range(n):
                j_subset = valid_subset[valid_subset[:,0] == j,2]
                if j_subset.any():
                    results[j,i] = np.sum(j_subset)

    results = softmax(results, axis=1)

    results = np.argmax(results, axis=1)

    return [group_mapping[i] for i in results]

def max_cohesion(cohesions, groups, n):
    group_list = np.unique(groups)
    results = np.zeros((n,len(group_list)))
    group_mapping = {}
    
    for i, g in zip(range(len(group_list)),group_list):
        group_mapping[i] = g
        span_set = (groups==g)[:,0]
        
        if span_set.any():
            valid_subset = cohesions[np.isin(cohesions[:,1],np.argwhere(span_set))]
            for j in range(n):
                j_subset = valid_subset[valid_subset[:,0] == j,2]
                if j_subset.any():
                    results[j,i] = np.max(j_subset)

    results = softmax(results, axis=1)

    results = np.argmax(results, axis=1)

    return [group_mapping[i] for i in results]

def median_cohesion(cohesions, groups, n):
    group_list = np.unique(groups)
    results = np.zeros((n,len(group_list)))
    group_mapping = {}
    
    for i, g in zip(range(len(group_list)),group_list):
        group_mapping[i] = g
        span_set = (groups==g)[:,0]
        
        if span_set.any():
            valid_subset = cohesions[np.isin(cohesions[:,1],np.argwhere(span_set))]
            for j in range(n):
                j_subset = valid_subset[valid_subset[:,0] == j,2]
                if j_subset.any():
                    results[j,i] = np.median(j_subset)

    results = softmax(results, axis=1)

    results = np.argmax(results, axis=1)

    return [group_mapping[i] for i in results]
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-d","--directory",dest="directory", help="Data directory", default=1000, type=str)
    args = parse.parse_args()

    pred = pd.read_csv(os.path.join(args.directory, 'test_data.csv'), header=None)
    data = pd.read_csv(os.path.join(args.directory, 'data.csv'), header=None)

    n = len(pred)

    cohesions = pd.read_csv('predicted-cohesions.csv').values
    cohesions[:,:2]-=1

    groups = pd.read_csv(os.path.join(args.directory, 'groups.csv'),header=None).values

    test_groups = pd.read_csv(os.path.join(args.directory, 'test_groups.csv'),header=None).values

    mean_cohesion_labels = mean_cohesion(cohesions, groups, n)
    print('Mean Cohesion Accuracy:', (np.array(mean_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

    #sns.set_palette("hls", 4)

    #plt.scatter(pd.concat([data,pred], axis=0).values[:,0],pd.concat([data,pred], axis=0).values[:,1], c=[1]*100 + [2]*100 + [2*i + 1 for i in mean_cohesion_labels], s=2)
    # df = pd.DataFrame({'X1' : pd.concat([data,pred], axis=0).values[:,0], 
    #                     'X2' : pd.concat([data,pred], axis=0).values[:,1],
    #                     'Color' : [str(i) for i in [1]*100 + [2]*100 + [i==cat for i in mean_cohesion_labels]],
    #                     'True Label' : [str(i) for i in [1]*100 + [2]*100 + [cat]*n]})
    # sns.scatterplot(x='X1', y='X2', hue='Color', style='True Label', data=df)
    # plt.title('Plot of Class ' + str(cat))
    # plt.show()

    total_cohesion_labels = total_cohesion(cohesions, groups, n)
    print('Total Cohesion Accuracy:', (np.array(total_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

    median_cohesion_labels = median_cohesion(cohesions, groups, n)
    print('Median Cohesion Accuracy:', (np.array(median_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

    max_cohesion_labels = max_cohesion(cohesions, groups, n)
    print('Max Cohesion Accuracy:', (np.array(max_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

    neigh3 = KNeighborsClassifier(n_neighbors=3)
    neigh3.fit(data,groups.reshape(-1))
    print('KNN 3 Neighbors Accuracy:', neigh3.score(pred, test_groups.reshape(-1)))

    neigh5 = KNeighborsClassifier(n_neighbors=5)
    neigh5.fit(data,groups.reshape(-1))
    print('KNN 5 Neighbors Accuracy:', neigh5.score(pred, test_groups.reshape(-1)))

    clf = LogisticRegression(multi_class='ovr').fit(data,groups.reshape(-1))
    print('Log Reg Accuracy:', clf.score(pred, test_groups.reshape(-1)))


    data_dict = {
                'Mean Accuracy' : [(np.array(mean_cohesion_labels)==test_groups.reshape(-1)).sum()/n],
                'Total Accuracy' : [(np.array(total_cohesion_labels)==test_groups.reshape(-1)).sum()/n],
                'Median Accuracy' : [(np.array(median_cohesion_labels)==test_groups.reshape(-1)).sum()/n],
                'Max Accuracy' : [(np.array(max_cohesion_labels)==test_groups.reshape(-1)).sum()/n],
                'KNN 3 Neighbors Accuracy' : [neigh3.score(pred, test_groups.reshape(-1))],
                'KNN 5 Neighbors Accuracy' : [neigh5.score(pred, test_groups.reshape(-1))],
                'Log Reg Accuracy' : [clf.score(pred, test_groups.reshape(-1))]     
                }

    if os.path.isfile('results.csv'):
        results_df = pd.read_csv('results.csv')
        data_dict_df = pd.DataFrame(data=data_dict)
        results_df = pd.concat([results_df,data_dict_df], axis=0)
    else:
        results_df = pd.DataFrame(data=data_dict)

    results_df.to_csv('results.csv', index=False)


from hmac import digest
import os
import numpy as np
import cupy as cp
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import softmax

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_digits, load_diabetes, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split


from pald_cupy_loop import pald

import argparse

def mean_cohesion(cohesions, groups, n):
    group_list = np.unique(groups)
    results = np.zeros((n,len(group_list)))
    group_mapping = {}
    
    for i, g in zip(range(len(group_list)),group_list):
        group_mapping[i] = g
        span_set = (groups==g)

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
        span_set = (groups==g)
        
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
        span_set = (groups==g)
        
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
        span_set = (groups==g)
        
        if span_set.any():
            valid_subset = cohesions[np.isin(cohesions[:,1],np.argwhere(span_set))]
            for j in range(n):
                j_subset = valid_subset[valid_subset[:,0] == j,2]
                if j_subset.any():
                    results[j,i] = np.median(j_subset)

    results = softmax(results, axis=1)

    results = np.argmax(results, axis=1)

    return [group_mapping[i] for i in results]
    
def num_strong(cohesions, groups, n):
    group_list = np.unique(groups)
    results = np.zeros((n,len(group_list)))
    group_mapping = {}
    
    bound = cohesions.trace()/(2*cohesions.shape[0])

    for i, g in zip(range(len(group_list)),group_list):
        group_mapping[i] = g
        span_set = (groups==g)
        
        if span_set.any():
            results[:,i] = np.sum((cohesions[:n,:-n] >= bound) & (groups == g).reshape(1,-1), axis=1)
            #results[:,i] = np.sum(cohesions[:-n,:][(cohesions[:-n,:] >= bound) & (groups == g).reshape(-1,1)], axis=0)

    results = softmax(results, axis=1)
    print(results)
    results = np.argmax(results, axis=1)

    return [group_mapping[i] for i in results]
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-d","--directory",dest="directory", help="Data directory", default=1000, type=str)
    args = parse.parse_args()


    # target_type = 'continuous'
    target_type = 'categorical'
    #digits = load_digits()
    # digits = load_breast_cancer()
    # digits = load_diabetes()
    print('Loading Data')
    digits = fetch_openml("vertebra-column", return_X_y=False, data_home='/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/mnist')
    #print(digits.keys())
    print('Splitting Data')
    data, pred, groups, test_groups = train_test_split(np.array(digits['data']), np.array(digits['target']), test_size=0.10, random_state=42)

    # pred = pd.read_csv(os.path.join(args.directory, 'test_data.csv'), header=None)
    # data = pd.read_csv(os.path.join(args.directory, 'data.csv'), header=None)
    # groups = pd.read_csv(os.path.join(args.directory, 'groups.csv'),header=None).values

    # test_groups = pd.read_csv(os.path.join(args.directory, 'test_groups.csv'),header=None).values

    # n = len(pred)

    print('Scaling Data')
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    pred = scaler.transform(pred)

    # cohesions = pd.read_csv('predicted-cohesions.csv').values
    # cohesions[:,:2]-=1
    print('Computing dist mat')
    dist_mat = squareform(pdist(data))

    print('Running PaLD')
    with cp.cuda.Device(1) as dev:
        cohesions = pald(dist_mat, dev=dev).get()

    # scaler = StandardScaler()
    # scaler = scaler.fit(data)
    # data = scaler.transform(data)
    # pred = scaler.transform(pred)

    #weights = np.sum(cohesions>=(cohesions.trace()/(cohesions.shape[0]*2)), axis=1)
    print('Computing Weights')
    weights = np.sum(cohesions>=(cohesions.trace()/(cohesions.shape[0]*2)), axis=0)
    #weights = np.sum(cohesions-np.identity(cohesions.shape[0])*np.diagonal(cohesions), axis=1)

    # mean_cohesion_labels = mean_cohesion(cohesions, groups, n)
    # print('Mean Cohesion Accuracy:', (np.array(mean_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

    # #sns.set_palette("hls", 4)

    # #plt.scatter(pd.concat([data,pred], axis=0).values[:,0],pd.concat([data,pred], axis=0).values[:,1], c=[1]*100 + [2]*100 + [2*i + 1 for i in mean_cohesion_labels], s=2)
    # # df = pd.DataFrame({'X1' : pd.concat([data,pred], axis=0).values[:,0], 
    # #                     'X2' : pd.concat([data,pred], axis=0).values[:,1],
    # #                     'Color' : [str(i) for i in [1]*100 + [2]*100 + [i==cat for i in mean_cohesion_labels]],
    # #                     'True Label' : [str(i) for i in [1]*100 + [2]*100 + [cat]*n]})
    # # sns.scatterplot(x='X1', y='X2', hue='Color', style='True Label', data=df)
    # # plt.title('Plot of Class ' + str(cat))
    # # plt.show()

    # total_cohesion_labels = total_cohesion(cohesions, groups, n)
    # print('Total Cohesion Accuracy:', (np.array(total_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

    # median_cohesion_labels = median_cohesion(cohesions, groups, n)
    # print('Median Cohesion Accuracy:', (np.array(median_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

    # max_cohesion_labels = max_cohesion(cohesions, groups, n)
    # print('Max Cohesion Accuracy:', (np.array(max_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

    print('Running Prediction Methods')
    if target_type == 'continuous':
        clf = LinearRegression().fit(data,groups.reshape(-1))
        print('Lin Reg R^2:', clf.score(pred, test_groups.reshape(-1)))

        clf = LinearRegression().fit(data,groups.reshape(-1), sample_weight=weights)
        print('Weighted Lin Reg R^2:', clf.score(pred, test_groups.reshape(-1)))
        
        clf = SVR(gamma='auto').fit(data,groups.reshape(-1))
        print('SVR R^2:', clf.score(pred, test_groups.reshape(-1)))

        clf = SVR(gamma='auto').fit(data,groups.reshape(-1), sample_weight=weights)
        print('Weighted SVR R^2:', clf.score(pred, test_groups.reshape(-1)))
        
    else:
        n = test_groups.reshape(-1).shape[0]
        print('Running KNN 3 Neighbors')
        neigh3 = KNeighborsClassifier(n_neighbors=3)
        neigh3.fit(data,groups.reshape(-1))
        print('KNN 3 Neighbors Accuracy:', neigh3.score(pred, test_groups.reshape(-1)))

        print('Running KNN 5 Neighbors')
        neigh5 = KNeighborsClassifier(n_neighbors=5)
        neigh5.fit(data,groups.reshape(-1))
        print('KNN 5 Neighbors Accuracy:', neigh5.score(pred, test_groups.reshape(-1)))

        print('Running Log Reg')
        clf = LogisticRegression(multi_class='ovr').fit(data,groups.reshape(-1))
        print('Log Reg Accuracy:', clf.score(pred, test_groups.reshape(-1)))

        print('Running Weighted Log Reg')
        clf = LogisticRegression(multi_class='ovr').fit(data,groups.reshape(-1), sample_weight=weights)
        print('Weighted Log Reg Accuracy:', clf.score(pred, test_groups.reshape(-1)))
        
        print('Running SVC')
        clf = SVC(gamma='auto').fit(data,groups.reshape(-1))
        print('SVC Accuracy:', clf.score(pred, test_groups.reshape(-1)))

        print('Running Weighted SVC')
        clf = SVC(gamma='auto').fit(data,groups.reshape(-1), sample_weight=weights)
        print('Weighted SVC Accuracy:', clf.score(pred, test_groups.reshape(-1)))

        print('Running Decision Tree')
        clf = DecisionTreeClassifier(random_state=0).fit(data,groups.reshape(-1))
        print('Decision Tree Accuracy:', clf.score(pred, test_groups.reshape(-1)))

        print('Running Weighted Decision Tree')
        clf = DecisionTreeClassifier(random_state=0).fit(data,groups.reshape(-1), sample_weight=weights)
        print('Weighted Decision Tree Accuracy:', clf.score(pred, test_groups.reshape(-1)))

        print('Computing dist mat')
        dist_mat = squareform(pdist(np.vstack([data, pred]), metric='hamming'))

        print('Running PaLD')
        with cp.cuda.Device(1) as dev:
            cohesions = pald(dist_mat, dev=dev).get()

        print('Running Total Cohesion')
        total_cohesion_labels = total_cohesion(cohesions, groups.reshape(-1), n)
        print('Total Cohesion Accuracy:', (np.array(total_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

        print('Running Median Cohesion')
        median_cohesion_labels = median_cohesion(cohesions, groups.reshape(-1), n)
        print('Median Cohesion Accuracy:', (np.array(median_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

        print('Running Max Cohesion')
        max_cohesion_labels = max_cohesion(cohesions, groups.reshape(-1), n)
        print('Max Cohesion Accuracy:', (np.array(max_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

        print('Running Mean Cohesion')
        mean_cohesion_labels = mean_cohesion(cohesions, groups.reshape(-1), n)
        print('Mean Cohesion Accuracy:', (np.array(mean_cohesion_labels)==test_groups.reshape(-1)).sum()/n)

        print('Running Sum Strong')
        num_strong_labels = num_strong(cohesions, groups.reshape(-1), n)
        print('Sum Strong Accuracy:', (np.array(num_strong_labels)==test_groups.reshape(-1)).sum()/n)
        print(num_strong_labels)
        print(test_groups)


    # data_dict = {
    #             'Mean Accuracy' : [(np.array(mean_cohesion_labels)==test_groups.reshape(-1)).sum()/n],
    #             'Total Accuracy' : [(np.array(total_cohesion_labels)==test_groups.reshape(-1)).sum()/n],
    #             'Median Accuracy' : [(np.array(median_cohesion_labels)==test_groups.reshape(-1)).sum()/n],
    #             'Max Accuracy' : [(np.array(max_cohesion_labels)==test_groups.reshape(-1)).sum()/n],
    #             'KNN 3 Neighbors Accuracy' : [neigh3.score(pred, test_groups.reshape(-1))],
    #             'KNN 5 Neighbors Accuracy' : [neigh5.score(pred, test_groups.reshape(-1))],
    #             'Log Reg Accuracy' : [clf.score(pred, test_groups.reshape(-1))]     
    #             }

    # if os.path.isfile('results.csv'):
    #     results_df = pd.read_csv('results.csv')
    #     data_dict_df = pd.DataFrame(data=data_dict)
    #     results_df = pd.concat([results_df,data_dict_df], axis=0)
    # else:
    #     results_df = pd.DataFrame(data=data_dict)

    # results_df.to_csv('results.csv', index=False)


import pandas as pd
import numpy as np
import time
import math
import argparse

import pprint
import os

from pald_numpy import pald as pald_numpy

from scipy.spatial.distance import pdist, squareform

# Set model parameters
interactive = False
t = 5

if __name__ == '__main__':
    data_dict = {
                'indexVal' : [],
                'mean1' : [],
                'std1' : [],
                'n1' : [],
                'dim1' : [],
                'dist1' : [],
                'mean2' : [],
                'std2' : [],
                'n2' : [],
                'dim2' : [],
                'dist2' : [],
                'time_pald_numpy' : [],
                'bound_pald_numpy' : [],
            }

    i=0       

    for dist1 in ['normal', 'exponential', 'chisquare']:
        for mean1 in [1]:
            for std1 in range(1, 80, 20):
                for dim in range(1,22,10):
                    for dist2 in ['normal', 'exponential', 'chisquare']:
                        for mean2 in range(1,200,75):
                            for std2 in range(1, 80, 20):
                                for n2 in range(10, 1001, 100):
                                    for n1 in range(10, 1001, 100):
                                        for num_times in range(t):
                                            i = i+1
                                            
                                            if (int(os.environ['SLURM_ARRAY_TASK_ID'])-1) == i%1000:
                                                data = np.zeros((n1+n2, dim))
                                                for d in range(dim):
                                                    if dist1 == 'normal':
                                                        data[:n1,d] = np.random.normal(mean1, std1, n1)
                                                    elif dist1 == 'exponential':
                                                        data[:n1,d] = np.random.exponential(mean1, n1)
                                                        std1 = mean1
                                                    elif dist1 == 'chisquare':
                                                        data[:n1,d] = np.random.chisquare(mean1, n1)
                                                        std1 = math.sqrt(2*mean1)  

                                                for d in range(dim):
                                                    if dist2 == 'normal':
                                                        data[n1:,d] = np.random.normal(mean2, std2, n2)
                                                    elif dist2 == 'exponential':
                                                        data[n1:,d] = np.random.exponential(mean2, n2)
                                                        std2 = mean2
                                                    elif dist2 == 'chisquare':
                                                        data[n1:,d] = np.random.chisquare(mean2, n2)
                                                        std2 = math.sqrt(2*mean2) 
                                                                    
                                                dist_mat = squareform(pdist(pd.DataFrame(data)))
                                                
                                                start_time = time.time()
                                                cohesions = pald_numpy(dist_mat)
                                                end_time = time.time()
                                                if interactive == True:
                                                        print('NumPy Normal')
                                                        print('Bound:', cohesions.trace()/(dist_mat.shape[0]*2))
                                                        print("--- %s seconds ---" % (end_time - start_time))

                                                data_dict['time_pald_numpy'] = [end_time-start_time]
                                                data_dict['bound_pald_numpy'] = [cohesions.trace()/(dist_mat.shape[0]*2)]

                                                

                                                if interactive == True:
                                                        print('\n\n')

                                                data_dict['indexVal'] = [i]
                                                data_dict['mean1'] = [mean1]
                                                data_dict['std1'] = [std1]
                                                data_dict['dist1'] = [dist1]
                                                data_dict['n1'] = [n1]
                                                data_dict['dim1'] = [dim]
                                                data_dict['mean2'] = [mean2]
                                                data_dict['std2'] = [std2]
                                                data_dict['dist2'] = [dist2]
                                                data_dict['n2'] = [n2]
                                                data_dict['dim2'] = [dim]

                                                pd.DataFrame(data=data_dict).to_csv('runs/full_run_results_' + str(os.environ['SLURM_ARRAY_TASK_ID']) + '.csv', index=False, mode='a', header=False)

                                if dist2 in ['exponential', 'chisquare']:
                                    break

                if dist1 in ['exponential', 'chisquare']:
                    break



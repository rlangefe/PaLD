import numpy as np
import pandas as pd

config_dict = {
                'mean1' : [],
                'std1' : [],
                'n1' : [],
                'dim1' : [],
                'dist1' : [],
                'n_test1' : [],
                'mean2' : [],
                'std2' : [],
                'n2' : [],
                'dim2' : [],
                'n_test2' : [],
                'dist2' : []
            }

n_test1 = 50
n_test2 = 50

for dist1 in ['normal']:
    for mean1 in [0]:
        for std1 in range(1, 20, 5):
            for dim in range(1,22,10):
                for dist2 in ['normal']:
                    for mean2 in range(0,200,50):
                        for std2 in range(1, 20, 5):
                            for n2 in range(10, 350, 100):
                                for n1 in range(10, 350, 100):
                                    config_dict['mean1'].append(mean1)
                                    config_dict['std1'].append(std1)
                                    config_dict['dist1'].append(dist1)
                                    config_dict['n1'].append(n1)
                                    config_dict['n_test1'].append(n_test1)
                                    config_dict['dim1'].append(dim)
                                    config_dict['mean2'].append(mean2)
                                    config_dict['std2'].append(std2)
                                    config_dict['dist2'].append(dist2)
                                    config_dict['n2'].append(n2)
                                    config_dict['n_test2'].append(n_test2)
                                    config_dict['dim2'].append(dim)
pd.DataFrame(data=config_dict).to_csv('info.csv', index=False)
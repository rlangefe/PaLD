import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('data.csv', header=None)
g1 = pd.read_csv('test_g1.csv', header=None)
g2 = pd.read_csv('test_g2.csv', header=None)



plt.scatter(pd.concat([data,g1,g2], axis=0).values[:,0],pd.concat([data,g1,g2], axis=0).values[:,1], c=[1]*100 + [2]*100 + [3]*100 + [4]*100, s=2)
plt.show()

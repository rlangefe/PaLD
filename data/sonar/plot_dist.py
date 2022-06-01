import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    num = 40
    df = pd.read_csv('matlab-results.csv')
    sns.histplot(x=df[(df['rows'] == num) | (df['cols'] == num)]['vals'])
    plt.savefig('cohesion_distribution.png')
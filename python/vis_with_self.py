import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('sonar_test.csv', header=None)
    groups = pd.read_csv('/deac/csc/classes/csc391_2/csc391/langrc18/PaLD/data/sonar/groups.csv', header=None)

    class_a = df.values[groups.values.reshape(-1) == 'M', groups.values.reshape(-1) == 'M']



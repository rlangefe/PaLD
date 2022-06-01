import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('sonar.all-data', header=None).values

X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.33, random_state=42)
pd.DataFrame(X_train).to_csv('data.csv', header=False, index=False)
pd.DataFrame(y_train).to_csv('groups.csv', header=False, index=False)

pd.DataFrame(X_test).to_csv('test_data.csv', header=False, index=False)
pd.DataFrame(y_test).to_csv('test_groups.csv', header=False, index=False)
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import sklearn

# 1. Read the data
ds = pd.read_csv('Data/Salary.csv')

# 2. Segregate the independent & dependent columns
X = ds.iloc[:, :-1].values
Y = ds.iloc[:, -1].values

# 3. Splitting dataset into the Training set and Test set. Test set size = 20%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
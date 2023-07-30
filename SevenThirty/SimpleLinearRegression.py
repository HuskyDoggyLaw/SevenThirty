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

# 4. Train the model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# 5. Predicting the test set results
Y_prediction = regressor.predict(X_test)
print (X_test)
print(Y_test)
print(Y_prediction)
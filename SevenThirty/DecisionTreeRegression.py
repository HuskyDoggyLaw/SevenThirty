import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import sklearn

datsetfilepath = "Data/Position_Salaries.csv"
datsetfileformat = "csv"
savemodelpath = "Data/linear_regression_model"

from Common.DataSetOperations import DataSetOperations
datasetoperations = DataSetOperations

from Common.ModelOperations import ModelOperations
modeloperations = ModelOperations

from sklearn.tree import DecisionTreeRegressor
trainmodel = DecisionTreeRegressor()
loadedmodel = DecisionTreeRegressor()

from sklearn.tree import export_text

# 1. Read the data
ds = datasetoperations.read(datsetfilepath, datsetfileformat)

# 2. Segregate the independent & dependent columns
X, Y = datasetoperations.segregate_xy(ds, 1, -1)

# 3. Splitting dataset into the Training set and Test set. Test set size = 20%
X_train, X_test, Y_train, Y_test = datasetoperations.split(X, Y, 0.2)

# 4. Train the model on the training set
trainmodel.fit(X_train, Y_train)

# 5. Predicting the test set results
Y_prediction = trainmodel.predict(X_test)

# 5.View the decision tree
mp.figure(figsize=(10, 8))
from sklearn.tree import plot_tree
feature_names = ['Position', 'Level', 'Salary']
plot_tree(trainmodel, feature_names=feature_names, filled=True)
mp.show()

print("X Test:")
print(X_test)
print("Y Test: ")
print(Y_test)
print("Y Prediction: ")
print(np.round(Y_prediction))

# 6. Read the user data and provide the prediction
while (0 < 1):
    userinput = input("Enter input: ")
    userinput2darray = np.array([userinput], dtype=np.float64).reshape(-1, 1)
    prediction = trainmodel.predict(userinput2darray)
    print ("Output: "+ str(np.round(prediction)))
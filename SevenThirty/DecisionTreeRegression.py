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

# 1. Read the data
ds = datasetoperations.read(datsetfilepath, datsetfileformat)

# 2. Segregate the independent & dependent columns
X, Y = datasetoperations.segregate_xy(ds, 1, -1)

# 3. Splitting dataset into the Training set and Test set. Test set size = 20%
X_train, X_test, Y_train, Y_test = datasetoperations.split(X, Y, 0.2)

# 4. Train the model on the training set
trainmodel.fit(X_train, Y_train)

# 4.a Optional - Serialize the model
modeloperations.serialize(trainmodel, savemodelpath)

# 5. Load the model from the serialized file and predicting the test set results
loadedmodel = modeloperations.load(savemodelpath)
Y_prediction = loadedmodel.predict(X_test)

# 5.a Print the expected & printed
print(X_test)
print(Y_test)
print(np.round(Y_prediction))

# 6. Read the user data and provide the prediction
while (0 < 1):
    userinput = input("Enter the years of experience: ")
    userinput2darray = np.array([userinput], dtype=np.float64).reshape(-1, 1)
    prediction = loadedmodel.predict(userinput2darray)
    print ("Salary: "+ str(np.round(prediction)))
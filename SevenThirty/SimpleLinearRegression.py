import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import sklearn

datsetfilepath = "Data/Salary2.csv"
datsetfileformat = "csv"
savemodelpath = "Data/linear_regression_model"

from Common.DataSetOperations import DataSetOperations
datasetoperations = DataSetOperations

from Common.ModelOperations import ModelOperations
modeloperations = ModelOperations

from sklearn.linear_model import LinearRegression
trainmodel = LinearRegression()
loadedmodel = LinearRegression()

from sklearn.metrics import accuracy_score

# Read the data
ds = datasetoperations.read(datsetfilepath, datsetfileformat)

# Segregate the independent & dependent columns
X, Y = datasetoperations.segregate_xy(ds,0,-1)

# Splitting dataset into the Training set and Test set. Test set size = 20%
X_train, X_test, Y_train, Y_test = datasetoperations.split(X, Y, 0.2)

# Train the model on the training set
trainmodel.fit(X_train, Y_train)

# Optional - Serialize the model
modeloperations.serialize(trainmodel, savemodelpath)

# Load the model from the serialized file and predicting the test set results
loadedmodel = modeloperations.load(savemodelpath)
Y_prediction = loadedmodel.predict(X_test)

# Print the results
print("X Test:", X_test)
print("Y Test: ", Y_test)
print("Y Prediction: ", np.round(Y_prediction))

# Read the input and provide the prediction
while (0 < 1):
    userinput = input("Enter the years of experience: ")
    userinput2darray = np.array([userinput], dtype=np.float64).reshape(-1, 1)
    prediction = loadedmodel.predict(userinput2darray)
    print ("Salary: "+ str(np.round(prediction)))
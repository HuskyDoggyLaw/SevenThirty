import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import sklearn

datsetfilepath = "Data/Disease2.csv"
datsetfileformat = "csv"

from Common.DataSetOperations import DataSetOperations
datasetoperations = DataSetOperations

from sklearn.tree import DecisionTreeClassifier
trainmodel = DecisionTreeClassifier()

from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Read the data
ds = datasetoperations.read(datsetfilepath, datsetfileformat)

# OneHotEncoding
ds_encoded = pd.get_dummies(ds, columns=['Gender', 'Blood Pressure', 'Smoking'])

# LabelEncoding for the Dependent Variable
label_encoder = LabelEncoder()
ds_encoded['Disease'] = label_encoder.fit_transform(ds_encoded['Disease'])

# Separating features and the dependent variable
X = ds_encoded.drop('Disease', axis=1)
Y = ds_encoded['Disease']

# Splitting dataset into the Training set and Test set. Test set size = 30%
X_train, X_test, Y_train, Y_test = datasetoperations.split(X, Y, 0.2)

# Train the model on the training set
trainmodel.fit(X_train, Y_train)

# Predicting the test set results
Y_prediction = trainmodel.predict(X_test)

# Measure the accuracy 
accuracy = accuracy_score(Y_test, Y_prediction)

# Print & view results
print("X Test:", X_test)
print("Y Test: ", Y_test)
print("Y Prediction: ", np.round(Y_prediction))
print("Accuracy: ", accuracy)
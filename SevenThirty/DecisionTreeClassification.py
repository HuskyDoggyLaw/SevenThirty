import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import sklearn

datsetfilepath = "Data/Disease.csv"
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

# Encode the columns both OneHot and Label encodings
hotcols = ['Gender', 'Smoking', 'Disease']
labelcols = ['Blood Pressure', 'Cholesterol', 'Blood Sugar', 'Diet']

preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), hotcols)], remainder='passthrough')
X = preprocessor.fit_transform(ds)

print(X)

label_encoder = LabelEncoder()
ds[labelcols] = ds[labelcols].apply(label_encoder.fit_transform)

print(X)

# Segregate the independent & dependent columns
X, Y = datasetoperations.segregate_xy(ds, 0, -1)


# Splitting dataset into the Training set and Test set. Test set size = 20%
X_train, X_test, Y_train, Y_test = datasetoperations.split(X, Y, 0.2)

# Train the model on the training set
trainmodel.fit(X_train, Y_train)

# Predicting the test set results
Y_prediction = trainmodel.predict(X_test)

# Measure the accuracy 
accuracy = accuracy_score(Y_test, Y_prediction)

#6. Print & view results
print("X Test:", X_test)
print("Y Test: ", Y_test)
print("Y Prediction: ", np.round(Y_prediction))
print("Accuracy: ", accuracy)

#View the decision tree
#mp.figure(figsize=(10, 8))
#from sklearn.tree import plot_tree
#feature_names = ['Input', 'Results']
#plot_tree(trainmodel, feature_names=feature_names, filled=True)
#mp.show()

# 7. Read the user data and provide the prediction
#while (0 < 1):
#    userinput = input("Enter input: ")
#    userinput2darray = np.array([userinput], dtype=np.float64).reshape(-1, 1)
#    prediction = trainmodel.predict(userinput2darray)
#    print ("Output: "+ str(np.round(prediction)))
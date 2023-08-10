import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

#Importing the dataset
filepath = "Data/Mall_Customers.csv"
df = pd.read_csv(filepath)
X = df.iloc[:, :].values
Y = df.iloc[:, -1].values

#Using the elbow method to find the optimal number of clusters

#Training the K-Means model on the datset

#Visualising the clusters
import numpy as np
import pandas as pd
import sklearn

class DataSetOperations:

     def __init__(self):
        self
     
     def read(fileloc, format):
         if format == "csv":
             return pd.read_csv(fileloc)

     def segregate_xy(dataset, y_col_start):
         return dataset.iloc[:, :y_col_start].values, dataset.iloc[:, y_col_start].values

     def split (X, Y, size):
         from sklearn.model_selection import train_test_split
         return train_test_split(X, Y, test_size=size, random_state=0)
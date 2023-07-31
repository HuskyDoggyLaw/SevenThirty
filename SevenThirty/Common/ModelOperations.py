import numpy as np
import pandas as pd
import sklearn
import joblib

class ModelOperations:

     def __init__(self):
        self
     
     def serialize(model, filename):       
       joblib.dump(model, filename)

     def load(filename):
         return joblib.load(filename)
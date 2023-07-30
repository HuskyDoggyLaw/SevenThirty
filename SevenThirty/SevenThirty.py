import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as mp

ds = pd.read_csv('C:/Users/manne/source/repos/SevenThirty/Data.csv')

x = ds.iloc[:, :-1].values
y = ds.iloc[:, -1].values

print (x)
print(y)

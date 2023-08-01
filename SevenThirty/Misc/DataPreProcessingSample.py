import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import sklearn

# 1. Read the data
print("Reading ...")
ds = pd.read_csv('Data/CarPurchase.csv')

# 2. Segregate the independent & dependent columns
X = ds.iloc[:, :-1].values
Y = ds.iloc[:, -1].values

# 3. Fill the missing data with the average value of the column 
print("Filling ...")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]) 

# 4. Encoding the independent data. Country Column will be broken down into 3 columns of 0s and 1s
print("Encoding ...")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 5. Encoding the independent data. Purchased Column Yes and No will be broken down into 0s and 1s
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

# 6. Splitting dataset into the Training set and Test set. Test set size = 20%
print("Splitting ...")
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# 7. Feature Scaling on the traing & test independent sets. 
# Scale only the columns that are NOT encoded in step 4
print("Feature scaling ...")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])

print (X_train)


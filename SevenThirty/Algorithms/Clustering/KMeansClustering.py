import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from sklearn.cluster import KMeans

#Importing the dataset
filepath = "Data/Mall_Customers.csv"
df = pd.read_csv(filepath)
X = df.iloc[:, :].values
Y = df.iloc[:, -1].values

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Add cluster assignments to the DataFrame
df['Cluster'] = labels

# Plot the data points and cluster centers
plt.scatter(df['Feature 1'], df['Feature 2'], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
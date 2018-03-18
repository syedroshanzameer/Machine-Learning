# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:53:46 2018

@author: Roshan Zameer Syed
"""

import numpy as np
import pandas as pd
data = pd.read_csv('Mall_Customers.csv')

#X = data[('Annual Income (k$)','Spending Score (1-100)')]
X = data.iloc[:,[3,4]].values

# elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i , init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 5,init='k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visualizing
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s=50,c = 'red',label = 'Cluster1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s=50,c = 'blue',label = 'Cluster2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s=50,c = 'green',label = 'Cluster3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s=50,c = 'cyan',label = 'Cluster4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s=50,c = 'magenta',label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200,c= 'red',marker = 'x',label = 'centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

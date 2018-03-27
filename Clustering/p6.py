# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 23:05:33 2018

@author: Roshan Zameer Syed
ID: 99999-2920
Description : Clustering using K-means(Elbow Method) and Hierarchical clustering 
"""
# Reading data from the dataset
import pandas as pd
data = pd.read_csv('3D_network.txt',header = None)
X = data.iloc[:,[1,3]].values

#K-means Elbow Method
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

# KMeans algorithm
kmeans = KMeans(n_clusters = 3,init='k-means++',random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the data
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s=50,c = 'red',label = 'Cluster1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s=50,c = 'blue',label = 'Cluster2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s=50,c = 'green',label = 'Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=50,c= 'red',marker = 'x',label = 'centroids')
plt.title('Clusters of customers')
plt.xlabel('Longitude')
plt.ylabel('Altitude')
plt.legend()
plt.show()

# Dendrogram to find optimal number of clusters
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucliden distance')
plt.show()

# Predicting the clustering results
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3 , affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizing the dataset 
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s=50,c = 'red',label = 'Cluster1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s=50,c = 'blue',label = 'Cluster2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s=50,c = 'green',label = 'Cluster3')
plt.title('Clusters of customers')
plt.xlabel('Longitude')
plt.ylabel('Altitude')
plt.legend()
plt.show()

"""
How many observations are in this dataset?
Ans : 1000

• How many clusters you got by using K-Means ? How many clusters you got by using
hierarchical clustering ? How you pick the number of clusters ?
Ans: K-means: I got 3 clusters using the elbow method
Hierarchical clustering: I got 3 clusters using this method
In the heirarchical clustering i selected the vertical line i.e largest Eulcidian distance between the clusters and got 3 clusters.
In the Elbow method the place where the elbow bends when referenced is 3.
"""
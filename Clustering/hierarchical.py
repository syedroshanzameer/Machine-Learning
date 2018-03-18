# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:29:08 2018

@author: Roshan Zameer Syed
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:53:46 2018

@author: Roshan Zameer Syed
"""

import pandas as pd
data = pd.read_csv('Mall_Customers.csv')

#X = data[('Annual Income (k$)','Spending Score (1-100)')]
X = data.iloc[:,[3,4]].values

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'single'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucliden distance')
plt.show()

from sklearn.cluster import AggolomerativeClustering
hc = AggolomerativeClustering(n_clusters = 5 , affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)



#Visualizing
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s=50,c = 'red',label = 'Cluster1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s=50,c = 'blue',label = 'Cluster2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s=50,c = 'green',label = 'Cluster3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s=50,c = 'cyan',label = 'Cluster4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s=50,c = 'magenta',label = 'Cluster5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:40:12 2018

@author: Roshan Zameer Syed
"""

import pandas as pd
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p=2)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_pred,y_test)

from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train

import numpy as np
x1 = np.arange(X_set[:,0].min(),X_set[:,0].max(),step = 0.1)
x2 = np.arange(X_set[:,1].min(),X_set[:,1].max(),step = 0.1)
X1,X2 = np.meshgrid(x1,x2)

Z = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)

import matplotlib.pyplot as plt
plt.contourf(X1,X2,Z.reshape(X1.shape),alpha = 0.55,
            cmap = ListedColormap(('red','blue')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j , 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','blue'))(j),label = j)

plt.title('KNN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test

import numpy as np
x1 = np.arange(X_set[:,0].min(),X_set[:,0].max(),step = 0.1)
x2 = np.arange(X_set[:,1].min(),X_set[:,1].max(),step = 0.1)
X1,X2 = np.meshgrid(x1,x2)

Z = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)

import matplotlib.pyplot as plt
plt.contourf(X1,X2,Z.reshape(X1.shape),alpha = 0.55,
            cmap = ListedColormap(('red','blue')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j , 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','blue'))(j),label = j)

plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:59:55 2018

@author: Roshan Zameer Syed
ID: 99999-2920
Description: Building models for classification KNN and SVM algorithms
"""
# Loading the IRIS dataset 
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target

# Splitting the dataset to Training & test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, 
                                                 random_state = 0)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Creating classifier KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p=2)
classifier.fit(X_train,y_train)

# Predicting test set results
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_pred,y_test)

# Accuracy score for  KNN
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred,y_test)

# Visualizing Training Set 
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train

import numpy as np
x1 = np.arange(X_set[:,0].min(),X_set[:,0].max(),step = 0.1)
x2 = np.arange(X_set[:,1].min(),X_set[:,1].max(),step = 0.1)
X1,X2 = np.meshgrid(x1,x2)
Z = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)

import matplotlib.pyplot as plt
plt.contourf(X1,X2,Z.reshape(X1.shape),alpha = 0.55,
            cmap = ListedColormap(('red','green','blue')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j , 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','green','blue'))(j),label = j)

# Plot title, label and legend
plt.title('KNN (Training set)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()


# Visualizing Testing Set
from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test

import numpy as np
x1 = np.arange(X_set[:,0].min(),X_set[:,0].max(),step = 0.1)
x2 = np.arange(X_set[:,1].min(),X_set[:,1].max(),step = 0.1)
X1,X2 = np.meshgrid(x1,x2)

Z = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)

import matplotlib.pyplot as plt
plt.contourf(X1,X2,Z.reshape(X1.shape),alpha = 0.55,
            cmap = ListedColormap(('red','green','blue')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j , 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','green','blue'))(j),label = j)

# Plot title, label and legend
plt.title('KNN (Test set)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Creating  SVM classifier
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear',random_state = 0)
classifier_svm.fit(X_train,y_train)
y_pred_svm = classifier_svm.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_svm = confusion_matrix(y_pred_svm,y_test)

# Accuracy score of SVM
from sklearn.metrics import accuracy_score
accuracy_svm = accuracy_score(y_pred_svm,y_test)


# Visualizing Training set - SVM
from matplotlib.colors import ListedColormap
X_set_svm,y_set_svm = X_train,y_train

import numpy as np
x1 = np.arange(X_set_svm[:,0].min(),X_set_svm[:,0].max(),step = 0.1)
x2 = np.arange(X_set_svm[:,1].min(),X_set_svm[:,1].max(),step = 0.1)
X1,X2 = np.meshgrid(x1,x2)
Z = classifier_svm.predict(np.array([X1.ravel(),X2.ravel()]).T)

import matplotlib.pyplot as plt
plt.contourf(X1,X2,Z.reshape(X1.shape),alpha = 0.55,
            cmap = ListedColormap(('red','green','blue')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for j in np.unique(y_set_svm):
    plt.scatter(X_set_svm[y_set_svm == j , 0], X_set_svm[y_set_svm == j, 1],
                c = ListedColormap(('red','green','blue'))(j),label = j)

# Plot title, label and legend
plt.title('SVM (Training set)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()


# Visualizing Testing Set - SVM
from matplotlib.colors import ListedColormap
X_set_svm,y_set_svm = X_test,y_test

import numpy as np
x1 = np.arange(X_set_svm[:,0].min(),X_set_svm[:,0].max(),step = 0.1)
x2 = np.arange(X_set_svm[:,1].min(),X_set_svm[:,1].max(),step = 0.1)
X1,X2 = np.meshgrid(x1,x2)

Z = classifier_svm.predict(np.array([X1.ravel(),X2.ravel()]).T)

import matplotlib.pyplot as plt
plt.contourf(X1,X2,Z.reshape(X1.shape),alpha = 0.55,
            cmap = ListedColormap(('red','green','blue')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for j in np.unique(y_set_svm):
    plt.scatter(X_set_svm[y_set_svm == j , 0], X_set_svm[y_set_svm == j, 1],
                c = ListedColormap(('red','green','blue'))(j),label = j)

# Plot title, label and legend
plt.title('SVM (Test set)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()


"""
• How many observations are in this data set ?
Ans: 150

• How many features are in this data set ?
Ans: 2

•Please compare the confusion matrix of both KNN and Linear SVM.Which algorithm
get a better confusion matrix ?

Ans: In the KNN confusion matrix the algorithm has predicted:
'0's - Correct labels 11  
'1's - Correct labels 5 , misclassifed 2 labels as 2
'2's - Correct labels 4 , misclassified 8 labels as 1
Accuracy = 66%

In the SVM confusion matrix the algorithm has predicted:
'0's - Correct labels 11  
'1's - Correct labels 8 , misclassifed 3 labels as 2
'2's - Correct labels 3 , misclassified 5 labels as 1
Accuracy = 73%

Both the algorithms has predicted couple of labels incorrect,
SVM is better as it has more number of accurate predictions with accuracy 73%


data = data.replace('?', np.NaN)                          # Replace missing data with NaN

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)    # Fill missing values with "Mean"
imp.fit(data)
data_clean = imp.transform(data)
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:06:46 2018

@author: Roshan Zameer Syed
ID: 99999-2920
Description: KNN and SVM implementation
"""

import pandas as pd
data = pd.read_csv('heartDisease.data',header=None)

#X_old = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
from sklearn.preprocessing import Imputer
import numpy as np
data = data.replace('?', np.NaN)

imp = Imputer(missing_values='NaN', strategy = 'mean',axis = 0)
imp.fit(data)
data_clean = imp.transform(data)
X = data_clean[:,:-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Knearest neighbor model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Confusion Matrix for KNN model 
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_pred,y_test)

# Accuracy Score for KNN model 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred,y_test)

# Linear SVM model 
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear',random_state = 0)
classifier_svm.fit(X_train,y_train)
y_pred_svm = classifier_svm.predict(X_test)

# Confusion Matrix for SVM model 
from sklearn.metrics import confusion_matrix
confusion_svm = confusion_matrix(y_pred_svm,y_test)

# Accuracy score fot SVM model 
from sklearn.metrics import accuracy_score
accuracy_svm = accuracy_score(y_pred_svm,y_test)

"""
How many observations are in this dataset
Ans: There are 294 Obervations

How many features?
Ans: There are 13 features in this dataset

Compare confusion matrix
Ans: The KNN confusion matrix predicted:
    0's - 28 correct and misrepresented 15
    1's - 11 correct and misrepresented 5
    
    The SVM confusion Matrix predicted:
    0's - 30 correct and misrepresented 7
    1's - 19 correct and misrepresented 3  
    
    The SVM is better as it has accuracy of 83%
"""

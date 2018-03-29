# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:06:36 2018

@author: Roshan Zameer Syed
"""
import pandas as pd
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13]
y = data.iloc[:,13].values

X = pd.get_dummies(X,columns=['Geography','Gender']).values

X = X[:,[0,8,9,11,1,2,3,4,5,6,7]]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing the kera labraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the Artificial Neural Network
classifier = Sequential()

#Adding the input layer and the 1st hidden layer
classifier.add(Dense(output_dim =6, init = 'uniform',activation = 'relu',input_dim = 11))

#Adding the second hidden layer
classifier.add(Dense(output_dim =6, init = 'uniform',activation = 'relu'))

#Adding the output Layer
classifier.add(Dense(output_dim =1, init = 'uniform',activation = 'sigmoid'))

# Compiling the artificial Neural Network
#classifier.compile(optimizer = 'sgd',loss= 'mean_squared_error',metrics = ['accuracy'])
classifier.compile(optimizer = 'adam',loss= 'mean_squared_error',metrics = ['accuracy'])

# Fitting the ANN to the Training Set
classifier.fit(X_train,y_train,batch_size=10,epochs=50)

#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
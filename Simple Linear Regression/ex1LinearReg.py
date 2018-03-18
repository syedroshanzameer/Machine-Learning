# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 07:12:01 2018

@author: Roshan Zameer Syed
"""

import pandas as pd
data = pd.read_csv('ex1data1.txt',header=None)
X = data.iloc[:,:1].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Population vs Profit (Training Set)')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in 10,000's")
plt.show()

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Population vs Profit (Test Set)')
plt.xlabel("Population in 10,000's")
plt.ylabel("Profit in 10,000's")
plt.show()
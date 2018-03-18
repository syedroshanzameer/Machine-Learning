# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:34:18 2018

@author: Roshan Zameer Syed
ID : 99999-2920
Description : Multivariate linear regression and backward elimination 
"""
# Reading the dataset
import pandas as pd
data = pd.read_csv('Advertising.csv')

# Feature and response matrix
X = data.iloc[:,[1,2,3]].values
y = data.iloc[:,-1].values

# Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

# Linear regresssion algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

ypred = regressor.predict(X_test)

import statsmodels.formula.api as sm
import numpy as np
# Adding new column of one's to X 
X = np.append(arr = np.ones((200,1)), values = X, axis = 1)

# Running Backward elimination algorithm
X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

import matplotlib.pyplot as plt
plt.scatter(X,y)
"""
• How many observations are in this data set ?
Ans : 200
• How many features are in this data set ?
Ans : 3 features
• What is the response for this data set ?
Ans : The last column sales is the response 
• Which predictors are the most significant for this dataset ? Please explain Why ?
Ans : Column 1- TV and column 2- Radio are the most significant predictors because 
their P-value is less than the threshold.
"""

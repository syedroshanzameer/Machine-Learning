# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:44:50 2018

@author: Roshan Zameer Syed
ID: 99999-2920
Desc: Multiple Linear Regression
"""

import pandas as pd
import numpy as np
data = pd.read_csv('HousePrice_UK.csv')
data = data.replace('?', np.NaN)
data_new = data.iloc[:,[3,4,9,22,23,44]].values
#X_n = data.iloc[:,[3,4,9,22,23]].values
#y = data.iloc[:,44].values

# Cleaning the dataset
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy = 'mean',axis = 0)
imp.fit(data_new)
data_clean = imp.transform(data_new)

X = data_clean[:,[0,1,2,3,4]]
y = data_clean[:,-1]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

y = (y-np.mean(y))/np.std(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

#from sklearn.preprocessing import StandardScaler
#sc1 = StandardScaler()
#y_train = sc1.fit_transform(y_train)
#y_test = sc1.transform(y_test)

#y_train_t = (y_train-np.mean(y_train))/np.std(y_train)
#y_test_t = (y_test-np.mean(y_test))/np.std(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


import statsmodels.formula.api as sm
import numpy as np
X = np.append(arr = np.ones((118818,1)), values = X, axis = 1)

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""
How many Observations?
Ans: there are 118818 observations

How many features?
Ans: 5 features

Which predictors are the most significant
Ans: 
"""


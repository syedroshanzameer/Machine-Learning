# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 06:40:54 2018

@author: Roshan Zameer Syed
ID : 99999-2920
Description : Building Linear regression model for Auto Insurance data
"""
# Reading the data set 
import pandas as pd
data = pd.read_excel('autoInsurance.xls')

#Slicing the dataset to X & y
X = data.iloc[:,:1].values
y = data.iloc[:,1].values

#Importing and spliting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#Building the linear regression model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualizing the training set results
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Claims vs payment (Training Set)')
plt.xlabel("Claims")
plt.ylabel("Payments")
plt.show()

# Visualizing the training set results
plt.scatter(X_test,y_test, color = "red")
plt.plot(X_train,regressor.predict(X_train), color = "blue")
plt.title("Claims vs Payment (Test set)")
plt.xlabel("Claims")
plt.ylabel("Payments")
plt.show()

"""
• How many observations are in this data set ?
Ans: 63 observations
• How many features are in this data set ?
Ans: X i.e column 1 is the feature. So, 1 feature.
• What is the response for this data set ?
Ans: Column Y = total payment for all the claims
"""
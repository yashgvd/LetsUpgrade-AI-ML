# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 07:28:36 2020

@author: gy605
"""

import pandas as pd 
import numpy as np 
dataset = pd.read_excel('Linear Regression.xlsx')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#print(dataset.isna().sum())
#print(dataset.describe())
#print(dataset.head())

X_s = pd.DataFrame(dataset["sqft_living"])
y = pd.DataFrame(dataset['price'])

xtrain,xtest, ytrain , ytest = train_test_split(X_s,y,test_size = .2, random_state = 2)

lin_reg = LinearRegression()

model_1 = lin_reg.fit(xtrain,ytrain)

ypred= lin_reg.predict(xtest)

from sklearn.metrics import mean_squared_error, r2_score

r2_s = r2_score(ytest, ypred)
mse_s = np.sqrt(mean_squared_error(ytest,ypred))

print(r2_s,mse_s)

X_b = pd.DataFrame(dataset["bedrooms"])

xtrain,xtest, ytrain , ytest = train_test_split(X_b,y,test_size = .2, random_state = 2)

lin_reg = LinearRegression()

model_1 = lin_reg.fit(xtrain,ytrain)

ypred= lin_reg.predict(xtest)

from sklearn.metrics import mean_squared_error, r2_score

r2_b = r2_score(ytest, ypred)
mse_b = np.sqrt(mean_squared_error(ytest,ypred))

print(r2_b,mse_b)

X_r = pd.DataFrame(dataset["bathrooms"])

xtrain,xtest, ytrain , ytest = train_test_split(X_r,y,test_size = .2, random_state = 2)

lin_reg = LinearRegression()

model_1 = lin_reg.fit(xtrain,ytrain)

ypred= lin_reg.predict(xtest)

from sklearn.metrics import mean_squared_error, r2_score

r2_r = r2_score(ytest, ypred)
mse_r = np.sqrt(mean_squared_error(ytest,ypred))

print(r2_r,mse_r)

X_f = pd.DataFrame(dataset["floors"])

xtrain,xtest, ytrain , ytest = train_test_split(X_f,y,test_size = .2, random_state = 2)

lin_reg = LinearRegression()

model_1 = lin_reg.fit(xtrain,ytrain)

ypred= lin_reg.predict(xtest)

from sklearn.metrics import mean_squared_error, r2_score

r2_f = r2_score(ytest, ypred)
mse_f = np.sqrt(mean_squared_error(ytest,ypred))

print(r2_f,mse_f)



'''
from sklearn.linear_model import LinearRegression
X = pd.DataFrame(dataset['sqft_living'], dataset['bedrooms'],dataset['bathrooms'],dataset['floors'])

xtrain,xtest, ytrain , ytest = train_test_split(X,y,test_size = .2, random_state = 2)

lin_reg = LinearRegression()

model_1 = lin_reg.fit(xtrain,ytrain)

ypred= lin_reg.predict(xtest)

from sklearn.metrics import mean_squared_error, r2_score

r2 = r2_score(ytest, ypred)
mse = np.sqrt(mean_squared_error(ytest,ypred))

print(r2,mse)
'''

features = ["sqft_living","bedrooms","floors","bathrooms"]

X = pd.DataFrame(dataset[features])

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()

scaled_X = scale.fit_transform(X)

dataset_scaled= pd.DataFrame(scaled_X)

xtrain,xtest, ytrain , ytest = train_test_split(scaled_X,y,test_size = .2, random_state = 2)


lin_reg = LinearRegression()

model_1 = lin_reg.fit(xtrain,ytrain)

ypred= lin_reg.predict(xtest)

from sklearn.metrics import mean_squared_error, r2_score

r2 = r2_score(ytest, ypred)
mse = np.sqrt(mean_squared_error(ytest,ypred))

print(r2)
print(mse)

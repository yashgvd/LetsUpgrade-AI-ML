# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:39:39 2020

@author: gy605
"""

import pandas as pd 
import numpy as np 
dataset = pd.read_excel('Linear Regression.xlsx')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_excel("Bank_Personal_loan_Modelling.xlsx",sheet_name = 1)

X = pd.DataFrame(dataset[['Age','Experience']])
y = pd.DataFrame(dataset['Personal Loan'])

import statsmodels.api as sm

X1 = sm.add_constant(X)

model_log = sm.Logit(y,X1)
result = model_log.fit()

print(result.summary())
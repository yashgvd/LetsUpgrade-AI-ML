# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:09:14 2020

@author: gy605
"""

import pandas as pd 
dataset = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx",sheet_name = 1)
dataset = dataset.drop({"ID","ZIP Code"},axis =1)
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()
from sklearn.ensemble import RandomForestClassifier
import numpy as np
dataset["CCAvg"] = np.round(dataset["CCAvg"])
rf_m = RandomForestClassifier(n_estimators = 1000, max_features = 2, oob_score= True)
features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account',
       'CD Account', 'Online', 'CreditCard']
result = rf_m.fit(X = dataset[features],y=dataset["Personal Loan"])
print(rf_m.oob_score_)

for feature,imp in zip(features,rf_m.feature_importances_):
    print(feature,imp)
    
from sklearn import tree 
tr_m = tree.DecisionTreeClassifier()
tr_m = tree.DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)
preds = pd.DataFrame([dataset["Education"],dataset["CCAvg"], dataset["Income"]]).T
tr_m.fit(X=preds,y=dataset["Personal Loan"])
with open("dtree_loan",'w') as f:
    f = tree.export_graphviz(tr_m,feature_names=["Education","CCAvg","Income"],out_file = f)
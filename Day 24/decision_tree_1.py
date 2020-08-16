# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:19:41 2020

@author: gy605
"""

import pandas as pd 
import numpy as np
from sklearn import preprocessing, tree

dataset = pd.read_csv('train.csv')

lab_enc = preprocessing.LabelEncoder()

dataset['Sex'] = lab_enc.fit_transform(dataset['Sex'])

model_t = tree.DecisionTreeClassifier(max_depth = 6)

X = pd.DataFrame(dataset[['Sex', 'Age', 'Fare']])
y = pd.DataFrame(dataset['Survived'])

result = model_t.fit(X,y)

with open('dtree1.dot','w') as f:
    f = tree.export_graphviz(model_t,feature_names=['Sex','Age','Fare'],out_file = f)
    

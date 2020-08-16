# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:23:50 2020

@author: gy605
"""

import pandas as pd               
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import neighbors
from sklearn import svm

dataset=pd.read_csv("train.csv")
le=preprocessing.LabelEncoder()
dataset["Sex"]=le.fit_transform(dataset["Sex"])
dataset["Embarked"]=le.fit_transform(dataset["Embarked"])
X=dataset.drop(["Pclass","PassengerId","Name","Ticket","Cabin"],axis=1)
y=dataset["Pclass"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
svm_model = svm.SVC(gamma=0.01,C=100)
svm_model.fit(X_train,y_train)

y_pred=svm_model.predict(X_test)
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

import numpy as np 

def svm_ac(features):
    a=[]
    b = []
    for feature in features:
        y = dataset[feature]
        X = dataset.drop([feature,"PassengerId","Name","Ticket","Cabin"],axis=1)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
        svm_model = svm.SVC(gamma = 0.01, C = 100)
        result = svm_model.fit(X_train,y_train)
        ypred = result.predict(X_test)
        Acc_value = accuracy_score(y_test,ypred,normalize = True)
        con_mat = confusion_matrix(y_test,ypred)
        a.append(Acc_value)
        b.append(con_mat)
    return features[np.argmax(a)], max(a),b[np.argmax(a)] 
features= ["Survived","Sex","Embarked","Parch","SibSp"]
f,a,c= svm_ac(features)
print("The feature ",f," gives the highest accuracy value of ",a," and the confusion matrix for this feature is : \n",c)




















# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:40:30 2020

@author: gy605
"""

import pandas as pd

dataset = pd.read_csv("train.csv")
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import *
le = preprocessing.LabelEncoder()
dataset["Sex"] = le.fit_transform(dataset["Sex"])
dataset["Embarked"] = le.fit_transform(dataset["Embarked"])
y=dataset["Survived"]
X=dataset.drop(["PassengerId","Survived","Name","Ticket","Cabin"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

clf_nb = BernoulliNB()
y_pred = clf_nb.fit(X_train,y_train).predict(X_test)

print(accuracy_score(y_test,y_pred,normalize=True))
print(confusion_matrix(y_test,y_pred))


features =["Pclass","Sex","SibSp","Parch","Embarked"]
for feature in features:
    y=dataset[feature]
    X=dataset.drop(["PassengerId",feature,"Name","Ticket","Cabin"],axis=1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    
    clf_nb = BernoulliNB()
    y_pred = clf_nb.fit(X_train,y_train).predict(X_test)
    Acc_value = accuracy_score(y_test,y_pred,normalize=True)
    conf_mat = confusion_matrix(y_test,y_pred)
    print("The accuracy value when DV is",features,"is",Acc_value)
    print("The confusion matrix when DV is",features, "is",conf_mat)


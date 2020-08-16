# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:54:12 2020

@author: gy605
"""

import pandas as pd               
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import neighbors

dataset=pd.read_csv("train.csv")
le=preprocessing.LabelEncoder()
dataset["Sex"]=le.fit_transform(dataset["Sex"])
dataset["Embarked"]=le.fit_transform(dataset["Embarked"])
X=dataset.drop(["Pclass","PassengerId","Name","Ticket","Cabin"],axis=1)
y=dataset["Pclass"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

knn_model = neighbors.KNeighborsClassifier(n_neighbors=3)
result =knn_model.fit(X_train,y_train)
print(result.score(X_test,y_test))

y_pred = knn_model.predict(X_test)
print(confusion_matrix(y_test,y_pred))

import numpy as np 

def knn_ac(y_test):
    a1=[]
    for k in range(1,len(y_test)+1):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
        test_model = neighbors.KNeighborsClassifier(n_neighbors=k)
        result = test_model.fit(X_train,y_train)
        Acc_value = result.score(X_test,y_test)
        a1.append(Acc_value)
    #print(a1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    test_model = neighbors.KNeighborsClassifier(n_neighbors=np.argmax(a1)+1)
    result = test_model.fit(X_train,y_train)  
    y_pred = test_model.predict(X_test)
    Conf_mat= confusion_matrix(y_test,y_pred)
    return max(a1), np.argmax(a1),Conf_mat 

a,k,c= knn_ac(y_test)
print("The best k value for this model is k = ",k+1,"which gives an accuracy of ",a, "and the confusion matrix for this value of k is : \n",c)








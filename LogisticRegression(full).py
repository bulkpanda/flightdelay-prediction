# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:45:27 2021

@author: Kunal Patel
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import chi2

dataset=pd.read_csv('FlightDelays.csv')

#=================Using all attributes======================================================
headings=['CRS_DEP_TIME','CARRIER','DEP_TIME','DEST','DISTANCE','FL_DATE','FL_NUM','ORIGIN','Weather','DAY_WEEK','DAY_OF_MONTH','TAIL_NUM']
used=headings

for i,j in enumerate(used):
    encode = pd.get_dummies(dataset[j])
    headers = encode.columns
    print(i,j, len(headers))

x=dataset[used].values #Independent variables
y=dataset.iloc[:,-1].values #Dependent variable



# Encoding categorical data
# Encoding the Independent Variable

le=LabelEncoder()
y=le.fit_transform(y)

x[:,-1]=le.fit_transform(x[:,-1])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,3,4,5,6,7,9,10,11])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
x=x.tolist()
x=x.todense()


#Spilliting data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1)

# Regularise
sc = StandardScaler()
X_train[:,(-2,-1)] = sc.fit_transform(X_train[:,(-2,-1)])
X_test[:,(-2,-1)] = sc.transform(X_test[:,(-2,-1)])

classifier= LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy=accuracy_score(y_test, y_pred)

print('Train accuracy :',classifier.score(X_train,y_train))
print('Test accuracy :',classifier.score(X_test,y_test))
print(classifier.coef_.shape)



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
from sklearn.feature_selection import SelectKBest

dataset=pd.read_csv('FlightDelays.csv')



headings=['CRS_DEP_TIME','CARRIER','DEP_TIME','DEST','DISTANCE','FL_DATE','FL_NUM','ORIGIN','Weather','DAY_WEEK','DAY_OF_MONTH','TAIL_NUM']
#============================================Model 3===================================================
used=['CRS_DEP_TIME','DEP_TIME','CARRIER','DEST','FL_NUM','ORIGIN','Weather','DAY_WEEK','DAY_OF_MONTH']
for i,j in enumerate(used):
    encode = pd.get_dummies(dataset[j])
    headers = encode.columns
    print(i,j, len(headers))
print('\n') 
x=dataset[used].values #Independent variables
y=dataset.iloc[:,-1].values #Dependent variable



# Encoding categorical data
# Encoding the Independent Variable

le=LabelEncoder()
y=le.fit_transform(y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2,3,4,5,6,7,8])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
x=x.tolist()
x=x.todense()


#Spilliting data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1)

# Regularise
sc = StandardScaler()
X_train[:,(-2,-1)] = sc.fit_transform(X_train[:,(-2,-1)])
X_test[:,(-2,-1)] = sc.transform(X_test[:,(-2,-1)])

#Train
classifier= LogisticRegression()
classifier.fit(X_train, y_train)

#Predict
y_pred = classifier.predict(X_test)

print('Train accuracy(3) :',classifier.score(X_train,y_train))
print('Test accuracy(4) :',classifier.score(X_test,y_test),'\n')


# chi sq analysis
target = pd.DataFrame(dataset['Flight Status']).to_numpy()
X = pd.DataFrame(dataset[used].replace(
    {'BWI':1,'DCA':2,'IAD':3,'JFK':4, 'LGA':5, 'EWR':6,'OH':7, 'DH':8, 'DL':9, 'MQ':10, 'UA':11, 'US':12, 'RU':13,
     'CO':14})).to_numpy()
chi22,pval = chi2(X,target)
feat = used
print('Chi sqaure:','\n')
for i in range(len(feat)):
    print(f'{feat[i]} & {chi22[i]:.3}')
print('\n')

# coefficient analysis  
print('Coefficients:','\n')     
print('Carriers:',classifier.coef_[0][0:8])
print('dest:',classifier.coef_[0][8:11])
print('fln:',classifier.coef_[0][11:114])
print('origin:',classifier.coef_[0][114:117])
print('weather:',classifier.coef_[0][117:119])
print('dayofweek:',classifier.coef_[0][119:126])
print('dayofweek:',classifier.coef_[0][119:126])
print('DOM:',classifier.coef_[0][126:157])
print('Time:',classifier.coef_[0][157:159])


#===============================Model 4=======================================================
print('\n', '==========Model 4=============', '\n')
used=['CRS_DEP_TIME','DEP_TIME','FL_NUM','Weather','DAY_WEEK','DAY_OF_MONTH']
for i,j in enumerate(used):
    encode = pd.get_dummies(dataset[j])
    headers = encode.columns
    print(i,j, len(headers))
print('\n') 
x=dataset[used].values #Independent variables
y=dataset.iloc[:,-1].values #Dependent variable



# Encoding categorical data
# Encoding the Independent Variable

le=LabelEncoder()
y=le.fit_transform(y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2,3,4,5])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
x=x.tolist()
x=x.todense()


#Spilliting data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1)

# Regularise
sc = StandardScaler()
X_train[:,(-2,-1)] = sc.fit_transform(X_train[:,(-2,-1)])
X_test[:,(-2,-1)] = sc.transform(X_test[:,(-2,-1)])

#Train
classifier= LogisticRegression()
classifier.fit(X_train, y_train)

#Predict
y_pred = classifier.predict(X_test)

print('Train accuracy(4) :',classifier.score(X_train,y_train))
print('Test accuracy(4) :',classifier.score(X_test,y_test),'\n')


# chi sq analysis
target = pd.DataFrame(dataset['Flight Status']).to_numpy()
X = pd.DataFrame(dataset[used]).to_numpy()
chi22,pval = chi2(X,target)
X_new = SelectKBest(chi2, k=2).fit_transform(X, target)
feat = used
print('Chi sqaure:','\n')
for i in range(len(feat)):
    print(f'{feat[i]} & {chi22[i]:.3}')
print('\n')    

# coefficient analysis    
print('Coefficients:','\n')    
print('fln:',classifier.coef_[0][0:103])    
print('weather:',classifier.coef_[0][103:105])
print('dayofweek:',classifier.coef_[0][105:112])
print('DOM:',classifier.coef_[0][112:143])
print('Time:',classifier.coef_[0][143:])


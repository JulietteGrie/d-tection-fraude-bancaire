# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:08:40 2022

@author: Juliette
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data = pd.read_csv(r'C:\Users\Juliette\OneDrive\Documents\Projet Python\détection fraude\data.csv')


'''
checking if no null value
does not contain null value => no exception to manage further
'''
print(data.isnull().sum()) 


'''
pie chart of the different types of transactions
'''
df = data["type"].value_counts()
transactions = df.index

figure = plt.pie(df, 
             labels = transactions, 
             autopct='%.2f',
             colors = ('blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple'),
             radius = 2)


'''
correlation coefficients (from -1 to 1): 
    (for clarity purposes, the absolute value of the coefficient is considered)
    the closest it is from 1, the better the linear correlation is
    1: perfect linear correlation
    0: no linear correlation
'''
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))


'''
turns the variable 'isFraud' into binary variable in the aim of building a model (only recognize numbers)
'''
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())


'''
building a model
answer as binary (labelled) value => classifier decision tree
'''
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


'''
prediction example
'''
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))
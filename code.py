# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:08:40 2022

@author: Juliette
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data = pd.read_csv(r'C:\Users\Juliette\OneDrive\Documents\Projet Python\détection fraude\data.csv')

## checking if no null value
## doesn't contain null value => no exception to manage further
print(data.isnull().sum()) 

## pie chart of the different types of transactions
df = data["type"].value_counts()
transactions = df.index

figure = plt.pie(df, 
             labels = transactions, 
             autopct='%.2f',
             colors = ('blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple'),
             radius = 2)


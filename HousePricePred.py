# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:43:56 2018


"""
#test


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

################
#IMPORT THE DATA
################
base_dir = 'C:/Users/John/Documents/GitHub/HousePrices/'
input_dir = base_dir + 'input/'
output_dir = base_dir + 'output/' 
file_name='train.csv'
abt= pd.read_csv(os.path.join(input_dir,file_name), encoding='utf8')

#######################
#CREATE DUMMY VARIABLES
#######################
cat_list = list(abt.select_dtypes(include=['object']).columns)
#abt2= pd.get_dummies(abt, columns=["Street", "LotConfig" ], prefix=["ST", "LC"])
abt2= pd.get_dummies(abt, columns=cat_list, prefix=cat_list)


train=abt2.sample(frac=0.7,random_state=200)
test=abt2.drop(train.index)

train_y = train.loc[:,['SalePrice']]
train_x = train.drop('SalePrice', 1)

#########################
#NEXT: DEAL WITH NULLS
#########################
abt.isnull().sum(axis = 0)
#list only cols with nulls
#If over 1000 nulls just delete col
#If categoricals have nulls what happens with the dummy encoding? Unknown?
#If low number on an of the numerics , maybe some imputation.....




# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)


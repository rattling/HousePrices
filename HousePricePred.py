@@ -8,10 +8,12 @@ Created on Sun Apr 22 17:43:56 2018

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import math 

#Next#
#Compare RMSE on train to test to see if overfit or no
#Examine residuals to see if any correlation unaccounted for at single variale level
#Check and deal with multicollinearity
#Build interaction and polynomial variables for potential inclusion
#Investigate regualarized regression if overfit
#Would feature seletcion help? 
#Any other potentially useful preprocessing?
#EDA to view single variable correlations
#Outlier removal needed?
#Distributions need to be altered?
#Data needs to be normalized to same scale etc.?
#Does regression assume/require a normal distribution for independent variables for large samples? I dont think so
#Go back over Andrew Ng advice here and generally
#More to try: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/60953
#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#https://www.kaggle.com/mjbahmani/20-ml-algorithms-for-house-prices-prediction
#https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard the links in here look great too

################
#IMPORT THE DATA
@ -27,7 +29,10 @@ abt= pd.read_csv(os.path.join(input_dir,file_name), encoding='utf8')
#######################
cat_list = list(abt.select_dtypes(include=['object']).columns)
#abt2= pd.get_dummies(abt, columns=["Street", "LotConfig" ], prefix=["ST", "LC"])
abt2= pd.get_dummies(abt, columns=cat_list, prefix=cat_list)
abt2.isnull().sum(axis = 0).sort_values(ascending=False)
abt2['LotFrontage'].fillna(0, inplace=True)
abt2['GarageYrBlt'].fillna(abt2['GarageYrBlt'].mean(), inplace=True)
abt2['MasVnrArea'].fillna(abt2['MasVnrArea'].mean(), inplace=True)


train=abt2.sample(frac=0.7,random_state=200)
@ -35,20 +40,34 @@ test=abt2.drop(train.index)

train_y = train.loc[:,['SalePrice']]
train_x = train.drop('SalePrice', 1)

test_y = test.loc[:,['SalePrice']]
test_x = test.drop('SalePrice', 1)
#########################
#NEXT: DEAL WITH NULLS
#########################
abt.isnull().sum(axis = 0)
a = abt2.isnull().sum(axis = 0)
#list only cols with nulls
#If over 1000 nulls just delete col
#If categoricals have nulls what happens with the dummy encoding? Unknown?
#If low number on an of the numerics , maybe some imputation.....




# Create linear regression object
regr = linear_model.LinearRegression()
# Fit the regression
regr.fit(train_x, train_y)

#Make predictions on the test set
pred_y = np.asarray(regr.predict(test_x))

#Replace negative/zero values wtih 1 so can take log
pred_y[pred_y<=0] = 1
#Take log to calculate the competition metric RMSE of log values
log_pred_y = np.log(pred_y)

#Repeat for the test set
test_y = np.asarray(test_y)
log_test_y = np.log(test_y)

#Calculate the RMSE
rmse=np.sqrt(metrics.mean_squared_error(log_test_y, log_pred_y))

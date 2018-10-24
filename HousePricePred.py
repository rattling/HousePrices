import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import math 

#Next#
#Compare RMSE on train to test to see if overfit or no
#Examine residuals to see if any correlation unaccounted for at single variale level
#Check and deal with multicollinearity
#Build interaction and polynomial variables for potential inclusion
#Investigate regualarized regression if overfit - try different alphas
#Cross validate rather than test/train?
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

#####################################################################
#IMPORT THE DATA
input_dir = "C:\\Users\\John\\Documents\\GitHub\\HousePrices\\input\\"
output_dir = "C:\\Users\\John\\Documents\\GitHub\\HousePrices\\output\\"
train_file = "train.csv"
score_file = "test.csv"
scored_file = "scored.csv"
abt= pd.read_csv(os.path.join(input_dir,train_file), encoding='utf8')
score=pd.read_csv(os.path.join(input_dir,score_file), encoding='utf8')

train_objs_num = len(abt)
dataset = pd.concat(objs=[abt, score], axis=0)
dataset_preprocessed = pd.get_dummies(dataset)
abt2 = dataset_preprocessed[:train_objs_num]
score2 = dataset_preprocessed[train_objs_num:]


####################################################################
#cat_list = list(abt.select_dtypes(include=['object']).columns)
abt2.isnull().sum(axis = 0).sort_values(ascending=False)
abt2['LotFrontage'].fillna(0, inplace=True)
abt2['GarageYrBlt'].fillna(abt2['GarageYrBlt'].mean(), inplace=True)
abt2['MasVnrArea'].fillna(abt2['MasVnrArea'].mean(), inplace=True)


train=abt2.sample(frac=0.7,random_state=200)
test=abt2.drop(train.index)

train_y = train.loc[:,['SalePrice']]
train_x = train.drop('SalePrice', 1)

test_y = test.loc[:,['SalePrice']]
test_x = test.drop('SalePrice', 1)


# Create linear regression object
#regr = linear_model.LinearRegression()
regr = Ridge(alpha=10,normalize=True)
# Fit the regression
regr.fit(train_x, train_y)


###################################################################
#Make predictions on the training set
fit_y = np.asarray(regr.predict(train_x))

#Replace negative/zero values wtih 1 so can take log
fit_y[fit_y<=0] = 1
#Take log to calculate the competition metric RMSE of log values
log_fit_y = np.log(fit_y)

#Repeat for the train set
train_y = np.asarray(train_y)
log_train_y = np.log(train_y)

#Calculate the RMSE for train set
train_rmse=np.sqrt(metrics.mean_squared_error(log_train_y, log_fit_y))
###################################################################

###################################################################
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
test_rmse=np.sqrt(metrics.mean_squared_error(log_test_y, log_pred_y))
###################################################################

###################################################################
#Score data for competition
###################################################################
score2['LotFrontage'].fillna(0, inplace=True)
score2['GarageYrBlt'].fillna(score['GarageYrBlt'].mean(), inplace=True)
score2['MasVnrArea'].fillna(score['MasVnrArea'].mean(), inplace=True)
score2 = score2.fillna(0)
score2 = score2.drop('SalePrice', 1)

#num_only = score2.select_dtypes(include=numerics)
#num_only = score2.select_dtypes(exclude=['object'])
#score2.isnull().sum(axis = 0).sort_values(ascending=False)
score_y = np.asarray(regr.predict(score2))
score_y= pd.DataFrame(score_y)
result = pd.concat([score["Id"], score_y], axis=1, sort=False)
result = result.rename(columns={result.columns[1]: "SalePrice" })


result.to_csv(os.path.join(output_dir,scored_file), encoding='utf8')

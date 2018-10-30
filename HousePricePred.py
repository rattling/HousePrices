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
from sklearn import preprocessing

#####################################################################
#DATA COLLECTION
#####################################################################
#IMPORT THE DATA
input_dir = "C:\\Users\\John\\Documents\\GitHub\\HousePrices\\input\\"
output_dir = "C:\\Users\\John\\Documents\\GitHub\\HousePrices\\output\\"
train_file = "train.csv"
score_file = "test.csv"
scored_file = "scored.csv"
abt= pd.read_csv(os.path.join(input_dir,train_file), encoding='utf8')
score=pd.read_csv(os.path.join(input_dir,score_file), encoding='utf8')

#APPEND ALL THE DATA AND ENSURE ALL LEVELS MATCHING FOR CATEGORICALS
train_objs_num = len(abt)
all_data = pd.concat(objs=[abt, score], axis=0)
all_data2 = pd.get_dummies(all_data)
all_data2= all_data2.drop('Id', 1)
all_data2_y = pd.DataFrame(all_data2['SalePrice'])
all_data2_x = all_data2.drop('SalePrice', 1)

#####################################################################
#PRE-PROCESSING
#####################################################################

#Better fill numeric nulls before creating polynomial versions of them!
all_data2_x['LotFrontage'].fillna(0, inplace=True)
all_data2_x['GarageYrBlt'].fillna(all_data2_x['GarageYrBlt'].mean(), inplace=True)
all_data2_x['MasVnrArea'].fillna(all_data2_x['MasVnrArea'].mean(), inplace=True)
all_data2_x = all_data2_x.fillna(0)

#Simpler way of doing it for now as can get col names etc and avoid huge num of vars created
num_list = list(all_data2_x.select_dtypes(exclude=['object']).columns)
for col_name in num_list:
    new_col_name = col_name + '^2'
    all_data2_x[new_col_name] = all_data2_x[col_name] **2
    #print (col_name)

#DONT THINK I NEED THIS AS THE REGRESSION NORMALIZES ANYWAY
#tmp = preprocessing.scale(all_data2_x)
#all_data3_x = pd.DataFrame(data=tmp, columns=all_data2_x.columns, index=all_data2_x.index)
#all_data3_x = all_data3_x.fillna(0)
all_data3_x = all_data2_x


# Get column names first
#names = all_data3_x.columns
# Create the Scaler object
#scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
#scaled_df = scaler.fit_transform(all_data3_x)
#scaled_df = pd.DataFrame(all_data3_x, columns=names)
    
abt2 = all_data3_x[:train_objs_num]
tmp_y= all_data2_y[:train_objs_num]
abt2['SalePrice']= tmp_y['SalePrice'].values
score2 = all_data3_x[train_objs_num:]

####################################################################

#####################################################################
#CREATE TRAIN/TEST DATASETS AND FIT A MODEL
#####################################################################
#abt2.isnull().sum(axis = 0).sort_values(ascending=False)

train=abt2.sample(frac=0.7,random_state=200)
test=abt2.drop(train.index)

train_y = train.loc[:,['SalePrice']]
train_x = train.drop('SalePrice', 1)

test_y = test.loc[:,['SalePrice']]
test_x = test.drop('SalePrice', 1)


def train_regression(x, y, alpha_param):
    regr = Ridge(alpha=alpha_param,normalize=True)
    regr.fit(x, y)
    return regr

def implement_regression(x, y,regr):
    #Create the regression object
    #Make predictions on the train data
    fit_y = np.asarray(regr.predict(x))
    #Replace negative/zero values wtih 1 so can take log
    fit_y[fit_y<=0] = 1
    #Take log to calculate the competition metric RMSE of log values
    log_fit_y = np.log(fit_y)    
    #Convert y aswell so can compare
    actual_y = np.asarray(y)
    log_actual_y = np.log(actual_y)    
    #Calculate the RMSE for train set
    rmse=np.sqrt(metrics.mean_squared_error(log_actual_y, log_fit_y))
    return rmse

print ("alpha, train_rmse, test_rmse")
mylist = [.1, .3, 1, 1.3, 1.6, 2.0, 3.0, 4.0, 5.0,6.0, 7.0, 8.0, 9.0, 10.0]
mylist = [2]
for alpha_param in mylist:
    regr=train_regression(train_x, train_y, alpha_param)
    train_rmse = implement_regression(train_x, train_y, regr)
    test_rmse = implement_regression(test_x, test_y, regr)
    print (str(alpha_param), str(train_rmse), str(test_rmse))
    
#print(regr.coef_)


###################################################################
#Score data for competition
###################################################################

#
score_y = np.asarray(regr.predict(score2))
score_y= pd.DataFrame(score_y)
result = pd.concat([score["Id"], score_y], axis=1, sort=False)
result = result.rename(columns={result.columns[1]: "SalePrice" })
result.to_csv(os.path.join(output_dir,scored_file), encoding='utf8')

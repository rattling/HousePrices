import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import math 
from sklearn import preprocessing as pp
from sklearn.model_selection import KFold
from scipy.stats import norm, skew #for some statistics


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


#####################################################################
#PRE-PROCESSING
#####################################################################

#APPEND ALL THE DATA AND ENSURE ALL LEVELS MATCHING FOR CATEGORICALS
train_objs_num = len(abt)
all_data = pd.concat(objs=[abt, score], axis=0)


#Deleting outliers
#all_data.drop(all_data[(all_data.GrLivArea>4000) & (all_data.SalePrice<300000)].index, inplace=True)
#all_data = all_data.reset_index()

all_data2= all_data.drop('Id', 1)
all_data2_y = pd.DataFrame(all_data2['SalePrice'])
all_data2_x = all_data2.drop('SalePrice', 1)
num_list = list(all_data2_x.select_dtypes(exclude=['object']).columns)

all_data2_x = pd.get_dummies(all_data2_x, drop_first=True)

#Better fill numeric nulls before creating polynomial versions of them!
all_data2_x['LotFrontage'].fillna(0, inplace=True)
all_data2_x['GarageYrBlt'].fillna(all_data2_x['GarageYrBlt'].mean(), inplace=True)
all_data2_x['MasVnrArea'].fillna(all_data2_x['MasVnrArea'].mean(), inplace=True)
all_data2_x = all_data2_x.fillna(0)

#Home grown polynomial feature generation
#for col_name in num_list:
#    new_col_name = col_name + '^2'
#    all_data2_x[new_col_name] = all_data2_x[col_name] **2
#    #print (col_name)

#More thorugh polynomial feature generation

poly = pp.PolynomialFeatures(2)
forPoly = all_data2_x[[c for c in all_data2_x.columns if c in num_list]]
notForPoly = all_data2_x[[c for c in all_data2_x.columns if c not in num_list]]
a1 = poly.fit_transform(forPoly)
a2 = poly.get_feature_names(forPoly.columns)
a3= pd.DataFrame(data=a1, columns = a2)
#For some reason the indexes gets garbled when you do the column subsetting
forPoly = forPoly.reset_index(drop=True)
notForPoly = notForPoly.reset_index(drop=True)
all_data3_x =  pd.concat([notForPoly, a3], axis=1)

##Check for skew and transform - EXPERIMENTAL
#For some reason doing this doubles my error! Not sure why
#num_list = list(all_data3_x.select_dtypes(exclude=['object']).columns)
#
#
## Check the skew of all numerical features
#skewed_feats = all_data3_x[num_list].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
#print("\nSkew in numerical features: \n")
#skewness = pd.DataFrame({'Skew' :skewed_feats})
#skewness.head(10)
#
#skewness = skewness[abs(skewness) > 0.75]
#print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
#
#from scipy.special import boxcox1p
#skewed_features = skewness.index
#lam = 0.15
#for feat in skewed_features:
#    #all_data[feat] += 1
#    all_data3_x[feat] = boxcox1p(all_data3_x[feat], lam)



# Now that data processed divide back out into train/test and scoring
abt2 = all_data3_x[:train_objs_num]
tmp_y= all_data2_y[:train_objs_num]
abt2['SalePrice']= tmp_y['SalePrice'].values
score2 = all_data3_x[train_objs_num:]

# Find most important features relative to target
print("Find most important features relative to target")
corr = abt2.corr()
corr['SalePriceAbs'] = corr['SalePrice'].abs()
corr.sort_values(["SalePriceAbs"], ascending = False, inplace = True)
print(corr.SalePriceAbs)
allVars = corr.index.tolist()
topVars = allVars[0:400]
abt2 = abt2[[c for c in abt2.columns if c  in topVars]]
score2 = score2[[c for c in score2.columns if c  in topVars]]

    

#####################################################################

#####################################################################
#CROSS VALIDATION AND FIT A MODEL
#####################################################################
#This lin ecost me hours! It is sorting the dataframe in place
#Which mucked up some join later on! Leaving it in as warning!
#abt2.isnull().sum(axis = 0).sort_values(ascending=False)

all_x = abt2.drop('SalePrice', 1)
all_y = abt2.loc[:,['SalePrice']]

###################################################################
def calc_error(X, y, model):
    '''returns in-sample error for already fit model.'''   
    predictions = np.asarray(model.predict(X))
    predictions[predictions<=0] = 1
    log_predictions= np.log(predictions)
    actual = np.asarray(y)    
    log_actual= np.log(actual)
    mse = mean_squared_error(log_actual, log_predictions)
    rmse = np.sqrt(mse)
    return rmse

def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the LRMSE for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_error(X_train, y_train, model)
    validation_error = calc_error(X_test, y_test, model)
    return train_error, validation_error

K = 2
kf = KFold(n_splits=K, shuffle=True, random_state=42)
alphas = [.1,.2, .25,.3,.35,.4, .5, 1, 1.3, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6,2.8, 3.0, 4.0, 5.0,6.0, 7.0, 8.0, 9.0, 10.0]
#alphas = [25, 26, 27, 28, 29,  30, 31, 32, 33, 34, 35]
#alphas = [2.0]

for alpha in alphas:
    train_errors = []
    validation_errors = []
    for train_index, val_index in kf.split(all_x, all_y):
        #print("Hello")
        #split data   
        X_train, X_val = all_x.iloc[train_index], all_x.iloc[val_index]
        y_train, y_val = all_y.iloc[train_index], all_y.iloc[val_index]
    
        # instantiate model
        model = Ridge(alpha=alpha,normalize=True)
                
        #calculate errors
        train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, model)
        
        # append to appropriate list
        train_errors.append(train_error)
        validation_errors.append(val_error)
    #    
    # generate report
    print('alpha: {:6} | mean(train_error): {:7} | mean(val_error): {}'.
          format(alpha,
                 round(np.mean(train_errors),4),
                 round(np.mean(validation_errors),4)))

###################################################################


###################################################################
#Score data for competition
###################################################################
#Do a final training using all data
alpha_param = .3
model = Ridge(alpha=alpha_param,normalize=True)
model.fit(all_x, all_y)
train_error = calc_error(all_x, all_y, model)   
score_y = np.asarray(model.predict(score2))
score_y[score_y<=0] = 1
score_y= pd.DataFrame(score_y)
result = pd.concat([score["Id"], score_y], axis=1, sort=False)
result = result.rename(columns={result.columns[1]: "SalePrice" })
#print(result.loc[result['SalePrice'] ==0])

result.to_csv(os.path.join(output_dir,scored_file), encoding='utf8', index=False)

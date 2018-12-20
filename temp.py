# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import linear_model
from sklearn import neighbors 
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing 
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from math import log
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from IPython.display import Image
from sklearn.externals.six import StringIO

house = pd.read_csv("kc_house_data.csv")
house = house.drop(['id', 'date'],axis=1)

str_list = []
for colname, colvalue in house.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)           
num_list = house.columns.difference(str_list) 
house_num = house[num_list]
f, ax = plt.subplots(figsize=(11, 11))
plt.title('Pearson Correlation of features')
sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)


#linear nodel
train_data, test_data = train_test_split(house, train_size = 0.8, random_state = 10)
def simple_linear_model(train, test, input_feature):
    regr = linear_model.LinearRegression() # Create a linear regression object
    regr.fit(train.as_matrix(columns = [input_feature]), train.as_matrix(columns = ['price'])) # Train the model
    RMSE = mean_squared_error(test.as_matrix(columns = ['price']), 
                              regr.predict(test.as_matrix(columns = [input_feature])))**0.5 # Calculate the RMSE on test data
    return RMSE, regr.intercept_[0], regr.coef_[0][0]

RMSE, w0, w1 = simple_linear_model(train_data, test_data, 'sqft_living')
print ('test error (RMSE) of linear model (sqft_living) is: %s ' %RMSE)


input_feature = train_data.columns.values.tolist()
input_feature.remove('price')

train_X = train_data.as_matrix(columns = input_feature)
scaler = preprocessing.StandardScaler().fit(train_X)
train_X_scaled = scaler.transform(train_X)
test_X = test_data.as_matrix(columns = [input_feature])
test_X_scaled = scaler.transform(test_X)

#KNN
knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
knn.fit(train_X_scaled, train_data.as_matrix(columns = ['price'])) 
print ('test error (RMSE) of KNN is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              knn.predict(test_X_scaled))**0.5) 

                    
#decision tree                    
clf = tree.DecisionTreeClassifier()
clf.fit(train_X_scaled, train_data.as_matrix(columns = ['price']))
print ('test error (RMSE) of decision tree is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              clf.predict(test_X_scaled))**0.5) 

#random foerest
clf1 = RandomForestClassifier(n_jobs=2)
clf1.fit(train_X_scaled, train_data.as_matrix(columns = ['price']))
print ('test error (RMSE) of random forest is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              clf1.predict(test_X_scaled))**0.5) # predict price and test error

#ridge regression  
clf3 = linear_model.Ridge(alpha = 1.0, normalize = True)
clf3.fit(train_data.as_matrix(columns = input_feature), train_data.as_matrix(columns = ['price']))
print ('test error (RMSE) of ridge is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              clf3.predict(test_data.as_matrix(columns = [input_feature])))**0.5) 
                 
#lasso regression
clf4 = linear_model.Lasso(alpha=0.1)
clf4.fit(train_data.as_matrix(columns = input_feature), train_data.as_matrix(columns = ['price']))
print ('test error (RMSE) of lasso is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              clf4.predict(test_data.as_matrix(columns = [input_feature])))**0.5) 

print(knn.predict(test_X_scaled))
print('tst')
print(test_data.as_matrix(columns = ['price']))
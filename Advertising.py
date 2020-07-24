# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:41:17 2020

@author: Izadi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
import os
os.chdir(r'D:\desktop\Python_DM_ML_BA\ML\LinearReg_ML')    # Dataset directory 
df = pd.read_csv("Advertising.csv", index_col=0)   # index_col=0 removing unnamed column

y = df.Sales
X = df.drop(['Sales'], axis=1)
#########################################################
# Feature importance
from xgboost import XGBRegressor
import xgboost
xgb =  XGBRegressor()
xgb.fit(X, y)
ax = xgboost.plot_importance(xgb, color='b') 
#########################################################df.shape
df.head()
df.info()
df.describe()
df.isnull().sum()   # Check if there is null values

# To make all data in one scale
dg = (df-df.mean())/df.std()
dg.describe()
import seaborn as sns
sns.distplot(y)           # is skewed to the right
sns.distplot(np.log(y))  # is skewed to the left
sns.distplot(np.log(y)+y/3)  # is nomal 
t#we take 
y = np.log(y)+y/3
y.isnull().sum()
# You can do the tansformations to the other variable, but I didn't.
# to check if there are highly correlated predictors.
sns.heatmap(df.corr('kendall'), annot=True, cmap='BuGn')
# From heatmap we see that the highes value is 0.49.
# So no need to drop any predictors. Only one can if is more than 0.85 or 0.9.
# Now to check how the regression line is doing with the predictors
sns.pairplot(dg, x_vars=['Internet', 'Email', 'Blog', 'WebBanner', 'Promotional', 'SmartPhone'],
             y_vars='Sales' , size=7, aspect=0.7 , kind='reg') 
plt.show()

from sklearn.linear_model import LinearRegression
estimator = LinearRegression()
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
features = X.columns.values
results = []
selector = RFE(estimator,3,step=1)   #recursive forward elimination 
selcetor = selector.fit(X,y)
selector.support_
# Out[61]: array([ True,  True,  True, False, False, False])
# It seems the first predictors are okay but not the the last three. 
# one can see if droping the last three will improve the model with all 6.
selector.ranking_
# Out[65]: array([2, 1, 3, 6, 5, 4])

for i in range(1,len(X.iloc[0])+1):
    selector = RFE(estimator, n_features_to_select=i, step=1)
    selector.fit(X,y)
    r2 = selector.score(X,y)
    selected_features = features[selector.support_]
    msr = mean_squared_error(y, selector.predict(X))
    results.append([i, r2, msr, ','.join(selected_features)])
    
results 

'''
results 
Out[68]: 
[[1, 0.47017552557905884, 2.5448877365932985, 'Email'],
 [2, 0.8987844810699489, 0.48616503259788457, 'Internet,Email'],
 [3, 0.9008606156394956, 0.47619280658599406, 'Internet,Email,Blog'],
 [4,
  0.9051564044419049,
  0.45555899148299284,
  'Internet,Email,Blog,SmartPhone'],
 [5,
  0.9081870005841042,
  0.4410022329163394,
  'Internet,Email,Blog,Promotional,SmartPhone'],
 [6,
  0.908406477352206,
  0.4399480276793689,
  'Internet,Email,Blog,WebBanner,Promotional,SmartPhone']]

'''
# Spliting the data into train and test 
np.random.seed(101) # for reproducibility of the same results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =  sc.fit_transform(X)
#split train and test sets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.28,random_state=0)

'''
Fist of all, I will do modeling with 4 regression in which they generate intercept and coefficients
estimations. In the next part I will do almost all regression models to find out the best one.

'''
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred =linreg.predict(X_test)
linreg.intercept_ 
linreg.coef_
a = pd.Series(linreg.intercept_)
ce1= pd.Series(linreg.coef_)

plt.figure(figsize=(10,10))
sns.regplot(y_pred, y_test, fit_reg=True, scatter_kws={"s": 100})

from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
ridge.intercept_
ridge.coef_

A = pd.concat([a,ce1], axis=0)
b = pd.Series(ridge.intercept_)
ce2= pd.Series(ridge.coef_)
B = pd.concat([b,ce2], axis=0)
d = pd.DataFrame({'A':A, 'B':B})
D = d.reset_index()
del D['index']
D

plt.figure(figsize=(10,10))
sns.regplot(y_pred, y_test, fit_reg=True, scatter_kws={"s": 100})

from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

                            

plt.figure(figsize=(10,10))
sns.regplot(y_pred, y_test, fit_reg=True, scatter_kws={"s": 100})

c = pd.Series(lasso.intercept_)
ce3 = pd.Series(lasso.coef_)
C = pd.concat([c,ce3], axis=0)
d1 = pd.DataFrame({'A':A, 'B':B, 'C':C})


from sklearn.linear_model import ElasticNet
en =  ElasticNet()
en.fit(X_train, y_train)
y_pred = en.predict(X_test)
en.intercept_  
en.coef_

plt.figure(figsize=(10,10))
sns.regplot(y_pred, y_test, fit_reg=True, scatter_kws={"s": 100})

from sklearn import svm
from sklearn.svm import SVR
svr = svm.SVR(kernel='linear', C=0.01)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
svr.intercept_  
svr.coef_

plt.figure(figsize=(10,10))
sns.regplot(y_pred, y_test, fit_reg=True, scatter_kws={"s": 100})


svr.coef = svr.coef_.flatten()
d = pd.Series(en.intercept_)
ce4 = pd.Series(en.coef_)
D = pd.concat([d,ce4], axis=0)
e = pd.Series(svr.intercept_)
ce5 = pd.Series(svr.coef)
E = pd.concat([e,ce5], axis=0)
dk = pd.DataFrame({'linear':A, 'ridge':B, 'lasso':C, 'elasticnet':D, 'svr':E})
dk.reset_index(drop=True, inplace=True)
dk
'''
dk
Out[159]: 
     linear     ridge     lasso  elasticnet       svr
0  4.071939  4.071941  4.068140    4.066878  4.052650
1  1.474936  1.473650  0.771592    0.794036  1.384780
2  1.017599  1.017242  0.268834    0.507164  1.062330
3  0.259153  0.259172  0.000000    0.000000  0.164303
4 -0.027676 -0.027109  0.000000    0.000000 -0.017284
5 -0.131404 -0.131408  0.000000    0.000000 -0.044582
6 -0.205768 -0.205511  0.000000    0.000000 -0.167716

Now we can fit all the models to find the champion one.
''''
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
from sklearn.linear_model import Lasso
lasso = Lasso()
from sklearn.linear_model import Ridge
ridge = Ridge()
from sklearn.linear_model import ElasticNet
en =  ElasticNet()
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
from sklearn.ensemble import AdaBoostRegressor
adr = AdaBoostRegressor()
import xgboost
from xgboost import XGBRegressor
xgb=XGBRegressor()
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
from sklearn.ensemble import BaggingRegressor 
br = BaggingRegressor() 
from catboost import CatBoostRegressor
cbr = CatBoostRegressor()
import lightgbm as ltb
lbm = ltb.LGBMRegressor()
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
from sklearn import svm
from sklearn.svm import SVR
svr = svm.SVR(kernel='linear', C=0.01)
from sklearn import svm
from sklearn.svm import SVR
clf = svm.SVR(C=1.0, kernel='poly', degree=2, gamma=2) 

models = [linreg, knr, ridge, lasso, en, dtr ,rfr, etr, adr, gbr, lbm, svr, clf, br, xgb]


model_names = ['LinearRegression',  'KNeighborsRegressor', 'Ridge', 'Lasso', 'ElasticNet',
               'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 
               'AdaBoostRegressor', 'GradientBoostingRegressor', 'ltb.LGBMRegressor', 'SVR', 
               'SVR','BaggingRegressor','XGBRegressor']
             

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    print("rmse of " + model_names[i] + ' is equal to {:.3f}'.format(np.sqrt(mean_squared_error(y_test, model.predict(X_test)))))  
    print('R-squared score of training ' + model_names[i] + ' is equal to {:.3f}'.format(model.score(X_train, y_train)))
    print('R-squared score of testing ' + model_names[i] + ' is equal to {:.3f}'.format(model.score(X_test, y_test)))


'''
rmse of LinearRegression is equal to 0.674
R-squared score of training LinearRegression is equal to 0.911
R-squared score of testing LinearRegression is equal to 0.902
rmse of KNeighborsRegressor is equal to 0.461
R-squared score of training KNeighborsRegressor is equal to 0.968
R-squared score of testing KNeighborsRegressor is equal to 0.954
rmse of Ridge is equal to 0.674
R-squared score of training Ridge is equal to 0.911
R-squared score of testing Ridge is equal to 0.902
rmse of Lasso is equal to 1.354
R-squared score of training Lasso is equal to 0.593
R-squared score of testing Lasso is equal to 0.605
rmse of ElasticNet is equal to 1.188
R-squared score of training ElasticNet is equal to 0.687
R-squared score of testing ElasticNet is equal to 0.696
rmse of DecisionTreeRegressor is equal to 0.434
R-squared score of training DecisionTreeRegressor is equal to 1.000
R-squared score of testing DecisionTreeRegressor is equal to 0.959
rmse of RandomForestRegressor is equal to 0.294
R-squared score of training RandomForestRegressor is equal to 0.998
R-squared score of testing RandomForestRegressor is equal to 0.981
rmse of ExtraTreesRegressor is equal to 0.261
R-squared score of training ExtraTreesRegressor is equal to 1.000
R-squared score of testing ExtraTreesRegressor is equal to 0.985
rmse of AdaBoostRegressor is equal to 0.577
R-squared score of training AdaBoostRegressor is equal to 0.937
R-squared score of testing AdaBoostRegressor is equal to 0.928
rmse of GradientBoostingRegressor is equal to 0.337
R-squared score of training GradientBoostingRegressor is equal to 0.990
R-squared score of testing GradientBoostingRegressor is equal to 0.976
rmse of ltb.LGBMRegressor is equal to 0.292
R-squared score of training ltb.LGBMRegressor is equal to 0.997
R-squared score of testing ltb.LGBMRegressor is equal to 0.982
rmse of SVR is equal to 0.688
R-squared score of training SVR is equal to 0.906
R-squared score of testing SVR is equal to 0.898
rmse of SVR is equal to 1.781
R-squared score of training SVR is equal to 0.386
R-squared score of testing SVR is equal to 0.317
rmse of BaggingRegressor is equal to 0.311
R-squared score of training BaggingRegressor is equal to 0.997
R-squared score of testing BaggingRegressor is equal to 0.979
rmse of XGBRegressor is equal to 0.341
R-squared score of training XGBRegressor is equal to 0.989
R-squared score of testing XGBRegressor is equal to 0.975
##########################################################################################################
DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor are doing the best but the rmse of
the RandomForestRegressor is the smallest. 
'''
##########################################################################################################################################


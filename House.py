# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:11:52 2022

@author: venki
"""

import pandas as pd
import numpy as np
# Load the data set of California Houses
df = pd.read_csv("D:\\House\\California_Houses.csv")

#Data understanding
df.info() ## information about the data type, null values and memory.

df.describe ## statistical information
df.shape # (20640, 14)
df.columns
"""
['Median_House_Value', 'Median_Income', 'Median_Age', 'Tot_Rooms',
       'Tot_Bedrooms', 'Population', 'Households', 'Latitude', 'Longitude',
       'Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego',
       'Distance_to_SanJose', 'Distance_to_SanFrancisco']"""
## value frequncy
dict = {}
for col in list(df.columns):
    dict[col] = df[col].value_counts().shape[0]
pd.DataFrame(dict, index = ["unique count"]).T

## Column types.

cont_col = ['Median_Income', 'Median_Age', 'Tot_Rooms', 'Tot_Bedrooms', 'Population', 'Households', 'Distance_to_coast', 
            'Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco']
loc_col = ['Latitude', 'Longitude']
dist_col = ['Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco']
target = 'Median_House_Value'

import matplotlib.pyplot as plt
import seaborn as sns
for col in df.columns:
    sns.boxplot(df[col]); plt.show()  ## we have a outliers

#Outlier treatment.

def remove_outlier(df, col_name):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    outlier = []
    for i in df[col_name]:
        if i<(q1 - 1.5 * iqr) or i>(q3 + 1.5 * iqr):
            outlier.append(i)
    outlier = pd.DataFrame(outlier,columns=['outlier'])
    print('There is {}% Outlier removed in {} according to IQR rule'.format(round((outlier.shape[0]/df.shape[0])*100,2),col_name))
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df = df[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df

def corrmatrix(df,target_col,num=len(df)):
    corrmatrix = df.corr()
    plt.figure(figsize = (20,10))
    columnss = corrmatrix.nlargest(num, target_col)[target_col].index
    cm = np.corrcoef(df[columnss].values.T)
    sns.set(font_scale = 1)
    hm = sns.heatmap(cm, cbar = True, annot = True, square = True, cmap = "RdPu" ,  fmt = ".2f", annot_kws = {"size": 10},
                     yticklabels = columnss.values, xticklabels = columnss.values)
    plt.show()

import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy import stats

def visualize_target(df, feature):
    print('Feature: {}, Skewness: {}, Kurtosis: {}'.format(feature,round(df[feature].skew(),5),round(df[feature].kurt(),5)))
    
    fig = plt.figure(constrained_layout=True, figsize=(12,6))
    grid = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)

    ax1 = fig.add_subplot(grid[0:2, :4])
    ax1.set_title('Histogram')
    sns.distplot(df.loc[:,feature], norm_hist=True,fit=norm, ax = ax1,color='indianred')

    ax2 = fig.add_subplot(grid[2:, :4])
    ax2.set_title('QQ_plot')
    stats.probplot(df.loc[:,feature], plot = ax2)

    ax3 = fig.add_subplot(grid[:, 4])
    ax3.set_title('Box Plot')
    sns.boxplot(y=df.loc[:,feature], orient='v', ax = ax3,color='indianred')

visualize_target(df,target)
# Feature: Median_House_Value, Skewness: 0.97776, Kurtosis: 0.32787

#Anomaly Detection and Removal

df = remove_outlier(df,target)

#There is 5.19% Outlier removed in Median_House_Value according to IQR rule

visualize_target(df,target)
# Feature: Median_House_Value, Skewness: 0.75595, Kurtosis: 0.0127

#Apply np.log() to target (Transformation)

df["Median_House_Value"] = np.log(df["Median_House_Value"])

visualize_target(df,target)
# Feature: Median_House_Value, Skewness: -0.31629, Kurtosis: -0.36984

# Check the distribution of continous variables

plt.figure(figsize=(20, 12))
for i, column in enumerate(cont_col, 1):
    plt.subplot(3, 4, i)
    sns.distplot(x=df[column],color='indianred')
    plt.legend()
    plt.xlabel(column)

# Correlation matrix
corrmatrix(df, target)


# Machine learning model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x = df.drop(['Median_House_Value','Longitude','Latitude'],axis=1)
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
y = df['Median_House_Value']

x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=0)
print(f'Train input: {x_train.shape}')
print(f'Train target: {y_train.shape}')
print(f'Test input: {x_test.shape}')
print(f'Test target: {y_test.shape}')

## fitting models

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn import ensemble
from xgboost.sklearn import XGBRegressor


r2 = []
model = []
RMSE = []
regressor = [LinearRegression(),RandomForestRegressor(),KNeighborsRegressor(),ensemble.GradientBoostingRegressor(),XGBRegressor()]
for obj in regressor:
    model_name = type(obj).__name__
    obj.fit(x_train,y_train)
    predict=obj.predict(x_test)
    r2.append(r2_score(y_test,predict))
    model.append(model_name)
    RMSE.append(mean_squared_error(y_test, predict, squared=False))
models = pd.DataFrame({'Model':model, 'r2':r2, 'RMSE':RMSE})
print(models)

models = models.set_index('Model')
plt.figure(figsize=(20,5))
models['RMSE'].sort_values().plot(kind='barh', color='r', align='center')
plt.title('RMSE for models')

reg = XGBRegressor()
reg.fit(x_train,y_train)
model_name = type(reg).__name__

for name, importance in zip(x, reg.feature_importances_):
    print(name, "=", round(importance,3))
    

features = x
importances = reg.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(20,5))
plt.title('Feature Importances of {}'.format(model_name))
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)),features.columns[indices])
plt.xlabel('Relative Importance')
plt.show()

#Median house value prediction

y_test = np.exp(y_test)
y_pred = np.exp(reg.predict(x_test))

model_name = type(reg).__name__
plt.subplots(figsize=(8,8))
plt.scatter(y_pred,y_test,s=4, color='indianred')
plt.plot(y_test, y_test,color='cornflowerblue', linewidth=3)
plt.title('{}: Predictions vs Observed Values'.format(model_name))
print("Accuracy score for {} is {:.2f}, RMSE is {:.2f}".format(model_name,r2_score(y_test, y_pred),mean_squared_error(y_test, y_pred, squared=False)))


import pickle

pickle.dump(reg, open('reg_model.pkl', 'wb'))

# Load the model from disk
model = pickle.load(open("reg_model.pkl", "rb"))

result = model.score(x_test, y_test)
print(result)


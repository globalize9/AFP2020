# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:33:55 2020

@author: yushi
"""

# example from https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
# test code for rf
import pandas as pd
import numpy as np

dataset = pd.read_csv('petrol_consumption.csv')
dataset.head()

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# preparing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the algo
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluation of the algo
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



#######################
# insert code to read the candidate pairs list
pair_names = 'AXTA US Equity_HXL US Equity'
pair_names = pair_names.split('_')

# reading in raw data, adapted from data_cleaning
all_data_2 = pd.read_excel("data_0719.xlsx", sheet_name = "Sheet5", header = [0,1])
all_data_2.index = all_data_2['Unnamed: 0_level_0']['Dates']
all_data_2_raw = all_data_2.drop(columns ='Unnamed: 0_level_0' )
all_data_2 = all_data_2_raw[all_data_2_raw.index <= "2019-12-31"] # specify end_date
price_data = all_data_2.xs('PX_LAST', axis = 1, level = 1, drop_level = False) # subsetting to PX_LAST only

# calculating the index as the ratio of PX_LAST
index_level = price_data[pair_names[0]] / price_data[pair_names[1]]
index_level = index_level.loc[macro_factors.index] # aligns the y data with the x data
index_level = index_level.dropna()

temp_macro = macro_factors.loc[index_level.index]
temp_feedstock = feedstock_factors.loc[index_level.index]

# replacing na's with 0s
temp_macro = temp_macro.fillna(0)
temp_feedstock = temp_feedstock.fillna(0)

# macrofactors test first
X = temp_macro.values
y = np.ravel(index_level.values)

# preparing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the algo
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluation of the algo
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


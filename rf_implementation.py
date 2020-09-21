# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:33:55 2020

@author: yushi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 
# import factors_cleaning # not sure the best way to connect the factors_cleaning script to this
# maybe copy/paste???

# import top 10 pairs
with open('top10.txt') as json_file:
    top10_pairs = json.load(json_file)

my_file = open('top10.txt')
content = my_file.read()
content.split(',')

import csv
with open('top10.txt'):
    
with open('top10.txt', newline = '') as games:                                                                                          
	game_reader = csv.reader(games, delimiter='\t')
	for game in game_reader:
		print(game)
    
# data is the list of pairs
# data.txt is the txt file you have created before to save the file into
import json
with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)


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

# we will drop all indicators that do not have 10 years of data completely
# forward fill on the remaining to close the NAs gap

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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

forest = RandomForestRegressor()
params = {'max_depth': [12, 15],
          'max_features': ['sqrt'],
          'min_samples_leaf': [10, 15, 20],
          'n_estimators': [150, 200, 250],
         'min_samples_split':[20, 25, 30]} #Using Validation Curves
time_series_split = TimeSeriesSplit(n_splits = 3)
forest_CV = RandomizedSearchCV(forest, params, n_iter = 25, cv = time_series_split, n_jobs = -1, verbose = 20)
forest_CV.fit(X_train, y_train)
y_pred = forest_CV.predict(X_test)
forest_CV.best_estimator_
forest_CV.best_params_


# Evaluation of the algo
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# =============================================================================
# # feature importances with forest of trees
# # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# importances = forest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
# 
# # Print the feature ranking
# print("Feature ranking:")
# 
# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# 
# # Plot the impurity-based feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#         color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()
# =============================================================================

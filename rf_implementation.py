# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:33:55 2020

@author: yushi
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import json

import os
# set path
os.chdir(r'C:\Users\yushi\Documents\GitHub\AFP2020')

# import factors_cleaning # runs the factors_cleaning.py code first

# =============================================================================
# # import factors_cleaning # not sure the best way to connect the factors_cleaning script to this
# 
# # import top 10 pairs
# with open('top10.json', 'r') as json_file:
#     top10_pairs = json.loads(json_file)
#     
# my_file = open('test10.txt')
# content = my_file.read()
# content.split(',')
# 
# import csv
# with open('top10.txt'):
#     
# with open('top10.txt', newline = '') as games:                                                                                          
# 	game_reader = csv.reader(games, delimiter='\t')
# 	for game in game_reader:
# 		print(game)
#     
# # data is the list of pairs
# # data.txt is the txt file you have created before to save the file into
# import json
# with open('data.txt', 'w') as outfile:
#     json.dump(data, outfile)
# =============================================================================

# Yushi hasn't figured out what to do with this portion yet...
# temporary solution
top10_pairs = ['AXTA US Equity_HXL US Equity','AVD US Equity_PRLB US Equity','AVD US Equity_TSE US Equity','AVD US Equity_DOW US Equity',
               'AVD US Equity_GCP US Equity','AVD US Equity_VVV US Equity','AVD US Equity_IMCD NA Equity','ADM US Equity_HXL US Equity',
               'AVD US Equity_WDFC US Equity','AVD US Equity_WLK US Equity','CINR US Equity_CE US Equity','GCP US Equity_OEC US Equity',
               'RPM US Equity_UNVR US Equity','ALB US Equity_CC US Equity','CRDA LN Equity_IMCD NA Equity','DSM NA Equity_OEC US Equity',
               'NZYMB DC Equity_OEC US Equity','CINR US Equity_OLN US Equity','KOP US Equity_MMM US Equity','MMM US Equity_OEC US Equity',
               'AXTA US Equity_KWR US Equity','GPRE US Equity_KWR US Equity','AXTA US Equity_DD US Equity','AXTA US Equity_MMM US Equity',
               'AXTA US Equity_HXL US Equity','GPRE US Equity_TSE US Equity','GPRE US Equity_PRLB US Equity','DCI US Equity_KWR US Equity',
               'KOP US Equity_TSE US Equity','AVD US Equity_TSE US Equity', 'AVD US Equity_BAS GR Equity','AXTA US Equity_NZYMB DC Equity',
               'AVD US Equity_CBT US Equity','GPRE US Equity_RPM US Equity','AVD US Equity_ADM US Equity']

# create a pairs_dict

lag_factors = pd.read_csv(r'C:\Users\yushi\Documents\GitHub\AFP2020\stepwise_factors.csv')
lag_factors.columns

def CheckDate(date_in,this_list):
    while date_in not in this_list:
        date_in -= datetime.timedelta(days = 1)
    return date_in

## move this section of code into the body

# need a day input, [last day of the training dataset]
# puting in an arbitrary date for now
input_date = datetime.date(2019,12,30)

nameP = top10_pairs[0]
row_num = np.where(nameP == lag_factors.pair)[0][0]
lag_factor_names = lag_factors.loc[row_num,['factor_1','factor_2','factor_3','factor_4','factor_5']]
lag_factor_times = lag_factors.loc[row_num,['lag_1','lag_2','lag_3','lag_4','lag_5']]
days_lag = lag_factor_times * 21 # converting to days equivalent
spec_dates = [input_date - datetime.timedelta(days = int(x)) for x in days_lag.tolist()]
# lag_dates = [CheckDate(x, index_level.index) for x in spec_dates]
# need2fix lag_dates
lag_dates = spec_dates

# =============================================================================
# X_vars = dict.fromkeys(lag_factor_names, np.NaN)
# 
# 
# for z in range(len(lag_factor_names)):
#     if lag_factor_names[z] in temp_macro.columns:
#         temp = temp_macro.loc[:,lag_factor_names[z]].copy()
#         temp = temp.shift(int(days_lag[z])) # takes care of the lag
#         temp = pd.DataFrame(temp, index = index_level.index)
#         X_vars[lag_factor_names[z]] = temp
#     else:
#         temp = temp_feedstock.loc[:,lag_factor_names[z]].copy()
#         temp = temp.shift(int(days_lag[z])) # takes care of the lag
#         X_vars[lag_factor_names[z]] = temp
# =============================================================================


X_vars = pd.DataFrame(np.NaN, index = index_level.index, columns = lag_factor_names.tolist())

# reading in the time series and lagging it by that much
for z in range(len(lag_factor_names)):
    if lag_factor_names[z] in temp_macro.columns:
        temp = temp_macro.loc[:,lag_factor_names[z]].copy()
        temp = temp.shift(int(days_lag[z])) # takes care of the lag
        X_vars.loc[:,lag_factor_names[z]] = temp
    else:
        temp = temp_feedstock.loc[:,lag_factor_names[z]].copy()
        temp = temp.shift(int(days_lag[z])) # takes care of the lag
        X_vars.loc[:,lag_factor_names[z]] = temp

X_vars = X_vars.dropna(axis = 0)

Y_var = index_level.shift(periods = -21*6) - index_level # 6 months forward

# matching the dates
joint_df = X_vars.merge(Y_var, how = 'outer', left_index = True, right_index = True)
last_day_X = joint_df.iloc[-1,:].drop('PX_LAST', axis = 0)
joint_df = joint_df.dropna()
features = joint_df.copy()
# switch to RF SD code

y = joint_df.iloc[:,-1]
X = joint_df.drop('PX_LAST', axis = 1)

        

# 0'th pair is same as 24

# factors are the actual values @ lag
# Y -> 6 periods ahead

for i in range(len(top10_pairs)):
    # insert code to read the candidate pairs list
    pair_names = top10_pairs[i]
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
    
    # temp_macro = factors_cleaning.macro_factors.loc[index_level.index]
    # temp_feedstock = factors_cleaning.feedstock_factors.loc[index_level.index]
    temp_macro = macro_factors.loc[index_level.index]
    temp_feedstock = feedstock_factors.loc[index_level.index]
    
    # we will drop all indicators that do not have 10 years of data completely
    # forward fill on the remaining to close the NAs gap
    
    # replacing na's with 0s
    temp_macro = temp_macro.fillna(0)
    temp_feedstock = temp_feedstock.fillna(0)
    
    ############ switch to RF_SD_3_sample_YW_mod.py #####################
    
    # macrofactors test first
    X = temp_macro
    y = np.ravel(index_level)
    
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
    
    forest.fit(X_train, y_train)
    forest.feature_importances_
    
    # Evaluation of the algo
    from sklearn import metrics
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# I think the importance plot is needed?
# https://machinelearningmastery.com/calculate-feature-importance-with-python/




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

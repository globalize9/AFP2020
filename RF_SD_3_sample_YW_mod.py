#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:14:11 2020

@author: sanchitadhirwani
"""

import pandas as pd
import numpy as np

features = pd.read_csv('train_PCA.csv')
features.head(5)
features = features.loc[features['Outlier Flag'] == 0] 
# Use numpy to convert to arrays

# labels are the values we want to predict
# Store Ids of houses
labels = np.array(features['target'])
Ids = np.array(features['ID'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('target', axis = 1)
features= features.drop('ID', axis = 1)
features = features.drop('Outlier Flag', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 500, stop = 1500, num = 50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(40, 100, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [50,60]
# Minimum number of samples required at each leaf node
min_samples_leaf = [5,10,15]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf,
               'bootstrap':bootstrap}
pprint(random_grid)

t_features = features
t_labels = labels
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=0, 
                               n_jobs = -1)
# Fit the random search model
rf_random.fit(t_features, t_labels)

rf_random.best_params_



# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [46,60,100],
    'max_features': ['auto'],
    'min_samples_leaf': [5,15],
    'min_samples_split': [45, 50, 55],
    'n_estimators': [806,850,1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 0)

grid_search.fit(t_features, t_labels)
grid_search.best_params_

best_grid = grid_search.best_estimator_

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    RMSE = np.sqrt(np.mean(np.square(abs(predictions - test_labels))))
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('RMS Error: {:0.4f} degrees.'.format(RMSE))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

grid_accuracy = evaluate (best_grid, test_features, test_labels)
grid_accuracy = evaluate (best_grid, train_features, train_labels)

def importance(model):
    fi = pd.DataFrame({'feature': feature_list,
                   'importance': model.feature_importances_})
    return fi

imp = importance (best_grid)

#Getting Predictions 
test_features_final = pd.read_csv('test_PCA_48.csv')
test_features_final.head(5)
Ids_test = np.array(test_features_final['ID'])
test_features_final= test_features_final.drop('ID', axis = 1)
model = best_grid
predictions_final = np.exp( model.predict(test_features_final))
submission =pd. DataFrame(np.array(list(zip(Ids_test, predictions_final))))
submission.to_csv('RF_out2.csv')








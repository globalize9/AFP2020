# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:26:12 2020

@author: yushi
"""

import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from itertools import combinations
from tabulate import tabulate

# set cwd
os.chdir("C:/Users/yushi/Documents/GitHub/AFP2020")

all_data_2 = pd.read_excel("Data_Full_Editable.xlsx", sheet_name = "Sheet3", header = [0,1])

all_data_2.index = all_data_2['Unnamed: 0_level_0']['Dates']

all_data_2 = all_data_2.drop(columns ='Unnamed: 0_level_0' )

companies = np.array(all_data_2.columns.get_level_values(0).unique()) #All the unique companies

all_combinations = []
for i in combinations(companies,2):
    all_combinations.append((i[0]+ "_" + i[1]))

all_combinations

df_1= pd.DataFrame({'Index':[],
                    'PE_Ratio':[],
                    'Div_Diff':[]})

###### calculating the technical indicators
df_1['Index'] = all_data_2[str(all_combinations[11].split('_')[0])]['PX_OFFICIAL_CLOSE']/all_data_2[str(all_combinations[11].split('_')[1])]['PX_OFFICIAL_CLOSE']
df_1['PE_Ratio'] = all_data_2[str(all_combinations[11].split('_')[0])]['BEST_PE_NXT_YR']/all_data_2[str(all_combinations[11].split('_')[1])]['BEST_PE_NXT_YR']
df_1['Div_Diff'] = (all_data_2[str(all_combinations[11].split('_')[0])]['AVERAGE_DIVIDEND_YIELD'] - all_data_2[str(all_combinations[11].split('_')[1])]['AVERAGE_DIVIDEND_YIELD'])*100



df_1['Return'] = df_1.Index.pct_change()

df_1['1Y_lag_Index'] = df_1['Index'].shift(252)

df_1['YoY_change'] = df_1['Index']/df_1['1Y_lag_Index'] - 1

df_1['1M_lag_Index'] = df_1['Index'].shift(21)
df_1['MoM_change'] = df_1['Index']/df_1['1M_lag_Index'] - 1

df_1['200_MA'] = df_1['Index'].rolling(window = 200).mean()
df_1['50_MA'] = df_1['Index'].rolling(window = 50).mean()

df_1['Up'] = df_1['Return']
df_1.loc[(df_1['Up']<0), 'Up'] = 0

df_1['Down'] = df_1['Return']
df_1.loc[(df_1['Down']>0), 'Down'] = 0 
df_1['Down'] = abs(df_1['Down']) # taking the absolute value for downward movements

df_1['avg_14up'] = df_1['Up'].rolling(window=14).mean()
df_1['avg_14down'] = df_1['Down'].rolling(window=14).mean()

df_1['RS'] = 100 - (100/(1+(df_1['avg_14up']/df_1['avg_14down'])))
df_1['RSI'] = 100 - (100/(1+ (df_1['avg_14up'].shift(1)*13 + df_1['Up'] + df_1['Down']) /
                          (df_1['avg_14down'].shift(1)*13 + df_1['Up'] + df_1['Down']) ))


df_1['200Day_diff'] = df_1['Index']/df_1['200_MA'] - 1



# picking an arbitrary start date...in this Dec 31 2018 
df_1.loc['2018-12-28']



# start = dt.datetime.strptime('2010-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')
df_1_test = df_1.loc[:end,]


indicators = pd.DataFrame({'Index':[0],
                           'RSI':[0],
                          'Std.dev_PE':[0],
                          'Avg_PE':[0],
                          'Recent_Peak':[0],
                          'MoM':[0],
                          '3M_MoM':[0],
                          'YoY':[0],
                          'Max_YoY':[0],
                          '50_MA':[0],
                          '200_MA':[0],
                          '50_day_test':[0],
                          'M-K_test':[0]})


start = dt.datetime.strptime('2018-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')
indicators['Index'] = df_1_test.loc['2018-12-31','Index']
indicators['RSI'] = df_1_test.loc['2018-12-31','RSI']
indicators['Std.dev_PE'] = df_1_test.loc[start:end,'PE_Ratio'].std()
indicators['Avg_PE'] = df_1_test.loc[start:end,'PE_Ratio'].mean()
indicators['Recent_Peak'] = 'LATER.'
indicators['MoM'] = df_1_test.loc['2018-12-31','MoM_change']
indicators['3M_MoM'] = df_1_test.loc['2018-10-01':'2018-12-31','MoM_change'].mean()
indicators['YoY'] = df_1_test.loc['2018-12-31','YoY_change']


df_1_test


indicators


start = dt.datetime.strptime('2018-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')
df_1.loc[start:end,'PE_Ratio'].std()



start = dt.datetime.strptime('2018-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')
plt.figure(figsize = (20,10))
df_1.loc[start:end,'Index'].plot()
df_1.loc[start:end,'200_MA'].plot(color="black")
df_1.loc[start:end, 'Index'].ewm(span = 20).mean().plot()



### visualizing the preliminary results

start = dt.datetime.strptime('2014-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2020-12-31', '%Y-%m-%d')

fig, ax = plt.subplots(2,1, figsize = (20,20))
ax[0].plot(df_1.loc[start:end,'Index'], color = 'red')
ax[1].plot(df_1.loc[start:end,'RSI'], color = 'blue')
ax[1].axhline(y=70, color='r')
ax[1].axhline(y=30, color='r')
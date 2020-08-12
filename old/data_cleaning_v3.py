#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from itertools import combinations

# set cwd
os.chdir("C:/Users/yushi/Documents/GitHub/AFP2020")

# In[3]:



all_data_2 = pd.read_excel("Data_Full_Editable.xlsx", sheet_name = "Sheet3", header = [0,1])

all_data_2.index = all_data_2['Unnamed: 0_level_0']['Dates']

all_data_2 = all_data_2.drop(columns ='Unnamed: 0_level_0' )

companies = np.array(all_data_2.columns.get_level_values(0).unique()) #All the unique companies

all_combinations = []
for i in combinations(companies,2):
    all_combinations.append((i[0]+ "_" + i[1]))

all_combinations


# In[231]:


df_1= pd.DataFrame({'Index':[],
                    'PE_Ratio':[],
                    'Div_Diff':[]})

###### calculating the technical indicators
df_1['Index'] = all_data_2[str(all_combinations[11].split('_')[0])]['PX_OFFICIAL_CLOSE']/all_data_2[str(all_combinations[11].split('_')[1])]['PX_OFFICIAL_CLOSE']
df_1['PE_Ratio'] = all_data_2[str(all_combinations[11].split('_')[0])]['BEST_PE_NXT_YR']/all_data_2[str(all_combinations[11].split('_')[1])]['BEST_PE_NXT_YR']
df_1['Div_Diff'] = (all_data_2[str(all_combinations[11].split('_')[0])]['AVERAGE_DIVIDEND_YIELD'] - all_data_2[str(all_combinations[11].split('_')[1])]['AVERAGE_DIVIDEND_YIELD'])*100


# In[232]:



df_1['Return'] = df_1.Index.pct_change()
df_1['Change'] = df_1.Index.diff()

df_1['1Y_lag_Index'] = df_1['Index'].shift(252)

df_1['YoY_change'] = df_1['Index']/df_1['1Y_lag_Index'] - 1

df_1['1M_lag_Index'] = df_1['Index'].shift(21)
df_1['MoM_change'] = df_1['Index']/df_1['1M_lag_Index'] - 1

df_1['200_MA'] = df_1['Index'].rolling(window = 200).mean()
df_1['50_MA'] = df_1['Index'].rolling(window = 50).mean()

df_1['Up'] = df_1['Change']
df_1.loc[(df_1['Up']<0), 'Up'] = 0

df_1['Down'] = df_1['Change']
df_1.loc[(df_1['Down']>0), 'Down'] = 0 
df_1['Down'] = abs(df_1['Down']) # taking the absolute value for downward movements

df_1['200Day_diff'] = df_1['Index']/df_1['200_MA'] - 1


# In[233]:


def get_RSI_14day(data):
    data.loc[:,'avg_14up'] = 0
    data.loc[:,'avg_14down'] = 0
    df_1_new = data.reset_index().copy()
    df_1_new.loc[13,'avg_14up'] = df_1_new.loc[range(0,14),'Up'].mean()
    df_1_new.loc[13,'avg_14down'] = df_1_new.loc[range(0,14),'Down'].mean()
    for i in range(14,df_1_new.shape[0]):
        df_1_new.loc[i,'avg_14up'] = (df_1_new.loc[(i-1),'avg_14up']*13 + df_1_new.loc[i,'Up'])/14
        df_1_new.loc[i,'avg_14down'] = (df_1_new.loc[(i-1),'avg_14down']*13 + df_1_new.loc[i,'Down'])/14
    df_1_new['RS'] = df_1_new['avg_14up']/df_1_new['avg_14down']
    df_1_new['RSI'] = (100 - (100/(1+df_1_new['RS'])))
    return(np.array(df_1_new['RSI']))


# In[234]:


df_1 = df_1.dropna()
df_1['RSI'] = get_RSI_14day(df_1)


# In[149]:


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


# In[337]:


start = dt.datetime.strptime('2018-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')
df_1.loc[start:end,'PE_Ratio'].std()



start = dt.datetime.strptime('2018-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')

fig = plt.figure(facecolor='white', figsize = (20,20))
ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
ax0.plot(df_1.loc[start:end,'Index'], color = 'blue')
ax0.plot(df_1.loc[start:end,'200_MA'])
ax0.plot(df_1.loc[start:end, 'Index'].ewm(span = 50).mean())
ax0.grid(True, color='black', linewidth = 0.2)
ax0.spines['bottom'].set_color("blue")
ax0.spines['top'].set_color("black")
ax0.spines['left'].set_color("black")
ax0.spines['right'].set_color("black")
plt.ylabel("200 day and 50 day Moving Average", fontsize = 12)
ax0.legend(("Index","200-Day Average","50-Day Average"), fontsize = 12)


### visualizing the preliminary results

start = dt.datetime.strptime('2018-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')

ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
ax1.plot(df_1.loc[start:end,'RSI'], color = 'blue')
ax1.axhline(y=70, color='r')
ax1.axhline(y=30, color='r')
plt.ylabel("Relative Strength Index", color = "black", fontsize = 12)
ax1.spines['bottom'].set_color("black")
ax1.spines['top'].set_color("black")
ax1.spines['left'].set_color("black")
ax1.spines['right'].set_color("black")
ax1.fill_between(df_1.loc[start:end,:].index, df_1.loc[start:end,'RSI'], 70, where=(df_1.loc[start:end,'RSI']>=70), facecolor='red', edgecolor='red', alpha=0.5)
ax1.fill_between(df_1.loc[start:end,:].index, df_1.loc[start:end,'RSI'], 30, where=(df_1.loc[start:end,'RSI']<=30), facecolor='green', edgecolor='green', alpha=0.5)
    
plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
plt.show()





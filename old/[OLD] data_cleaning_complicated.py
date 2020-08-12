# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:23:57 2020

@author: yushi
"""

import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# os.getcwd()
# 
# df = pd.read_csv("C:\Users\yushi\OneDrive - ualberta.ca\Applied Finance Project\Data\Half_Data.xls")
# df = pd.read_excel("Data_Full_Editable.xlsx", skiprows = 2)
# original_data = df
# 
# equity_names = df.iloc[0,:][df.iloc[0,:].notna()]   
# indicators = df.iloc[2,:]
# 
# y_dim = indicators.size
# x_dim = original_data.shape[0]
# 
# columns = pd.MultiIndex.from_product([list(equity_names), list(indicators)])
# df = pd.DataFrame(np.arange(y_dim*x_dim).reshape((x_dim, y_dim)), columns = columns)
# 
# df.to_flat_index()
# 
# df.columns.get_level_values[0]
# =============================================================================


## new attempt...reading in columns in chunks of 13
os.chdir(r"C:\Users\yushi\OneDrive - ualberta.ca\Applied Finance Project\Data")

which_col = np.hstack((1,np.arange(2,14)))
all_data = pd.read_excel("Data_Full_Editable.xlsx", skiprows = 2)

num_of_equities = int((all_data.shape[1]-1)/13)
names_indices = np.array([i*13+1 for i in range(num_of_equities)])

equity_names_all = all_data.iloc[0, names_indices]
temp = equity_names_all.str.split(" ")
equity_names = equity_names_all.copy()
for i in range(len(temp)):
    equity_names.iloc[i] = temp[i][0]

all_data = all_data.drop(all_data.index[0:2])
column_names = all_data.iloc[0,np.arange(1,14)]

list_eq = []
# list_eq is the comprehension database
# separating each of the equities into its own individual dataframe
for i in range(len(equity_names)-1):
    data_temp = all_data.iloc[2:, np.hstack((0,np.arange(names_indices[i],names_indices[i+1])))]
    data_temp.columns = np.hstack((equity_names[i] + " date",column_names))
    list_eq.append(data_temp)


# RSI indicator 
stock_temp = list_eq[1].copy()
stock_temp.insert(stock_temp.shape[1],"rsi_i", "NaN")
rsi_temp = stock_temp.loc[:,"RSI_14D"]

for i in range(len(rsi_temp)):
    if rsi_temp.iloc[i] >= 70: stock_temp.loc[i,"rsi_i"] = 1
    elif rsi_temp.iloc[i] <= 30: stock_temp.loc[i,"rsi_i"] = 0
    else: stock_temp.loc[i,"rsi_i"] = "NaN"

stock_temp = stock_temp[stock_temp.iloc[:,0].notna()] # drop dates with NA


list_eq[0]["PX_OFFICIAL_CLOSE"].plot()
list_eq[5]["PX_OFFICIAL_CLOSE"].plot()
list_eq[15]["PX_OFFICIAL_CLOSE"].plot()
list_eq[25]["PX_OFFICIAL_CLOSE"].plot()



# plotting
ax1 = plt.axes(stock_temp.iloc[:,0])
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('RSI')

ax1.plot(x = stock_temp.iloc[:,0], y = stock_temp.iloc[:,1])
ax2.plot(y = stock_temp.iloc[:,8])

datetime.strptime(stock_temp.iloc[:,0], )
stock_temp.iloc[:,0]


plt.figure()
plt.plot()
plt.legend(["EFD "+col_names[3])






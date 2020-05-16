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

os.getcwd()

df = pd.read_csv("C:\Users\yushi\OneDrive - ualberta.ca\Applied Finance Project\Data\Half_Data.xls")
df = pd.read_excel("Data_Full_Editable.xlsx", skiprows = 2)
original_data = df

equity_names = df.iloc[0,:][df.iloc[0,:].notna()]   
indicators = df.iloc[2,:]

y_dim = indicators.size
x_dim = original_data.shape[0]

columns = pd.MultiIndex.from_product([list(equity_names), list(indicators)])
df = pd.DataFrame(np.arange(y_dim*x_dim).reshape((x_dim, y_dim)), columns = columns)

df.to_flat_index()

df.columns.get_level_values(1)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ipynb.fs.full.Executions as executions
import pandas as pd


# # Read Data

# In[18]:


data = pd.read_excel("Final_Data.xlsx", sheet_name = "Sheet3", header = [0,1])
macro_factors_raw = pd.read_excel('BBGFactors.xlsx', sheet_name = 'macroFactors')
feedstock_factors_raw = pd.read_excel('BBGFactors.xlsx', sheet_name = 'feedstockFactors')


# # Execute File

# In[ ]:


executions.Execute(data, macro_factors_raw, feedstock_factors_raw)


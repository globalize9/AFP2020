# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os

# set path
os.chdir(r'C:\Users\yushi\Documents\GitHub\AFP2020')

# read in macro and feedstock factors
macro_factors_raw = pd.read_excel('BBGFactors.xlsx', sheet_name = 'macroFactors')
feedstock_factors_raw = pd.read_excel('BBGFactors.xlsx', sheet_name = 'feedstockFactors')

def CleanFactors(df, cut_off_date):
    # preps the df for further analysis, specify date in this format '2020-08-20'
    df.index = df.Dates
    date_loc = np.where(df.index == cut_off_date)[0]
    if len(date_loc) == 0: return 'Invalid date'
    date_loc = date_loc[0]
    df = df.loc[df.index[:date_loc]]
    df = df.drop('Dates', axis = 1)
    df = df.ffill()
    df = df.dropna(axis = 1)
    return df

cut_off_date = '2019-12-31'
macro_factors_round1 = CleanFactors(macro_factors_raw, cut_off_date)
feedstock_factors_round1 = CleanFactors(feedstock_factors_raw, cut_off_date)

# YoY adjustment
def YoYClean(factors_df):
    macro_names = factors_df.columns.tolist()
    pct_yoy = ['YoY' in x for x in macro_names]
    factors_df_adj = pd.DataFrame(np.NaN, index = factors_df.index, columns = macro_names)
    for i in range(len(macro_names)):
        if pct_yoy[i] == True:
            factors_df_adj.iloc[:,i] = (factors_df.iloc[:,i] / factors_df.iloc[:,i].shift(252) - 1) * 100
        else:
            factors_df_adj.iloc[:,i] = factors_df.iloc[:,i]
    
    factors_df_adj = factors_df_adj.dropna(axis = 0) # drop observations instead of columns this time
    return factors_df_adj

macro_factors = YoYClean(macro_factors_round1)
feedstock_factors = YoYClean(feedstock_factors_round1)
feedstock_factors = (feedstock_factors_round1 / feedstock_factors_round1.shift(252) - 1) * 100
feedstock_factors = feedstock_factors.dropna(axis = 0)


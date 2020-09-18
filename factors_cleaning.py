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
    df = df.ffill()
    return df

cut_off_date = '2019-12-31'
macro_factors = CleanFactors(macro_factors_raw, cut_off_date)
feedstock_factors = CleanFactors(feedstock_factors_raw, cut_off_date)


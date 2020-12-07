#!/usr/bin/env python
# coding: utf-8

# In[38]:


import ipynb.fs.full.All_Functions as All_Functions
import pandas as pd
import numpy as np


# # Final Function

# In[6]:


def Execute(data, macro_raw, feedstock_raw):
    
    #Clean data and get the price_data
    data.index = data['Unnamed: 0_level_0']['Dates']
    data = data.drop(columns ='Unnamed: 0_level_0' )
    data = data[data.index <= "2019-12-31"] # specify end_date

    price_data = data.xs('PX_LAST', axis = 1, level = 1, drop_level = True) # subsetting to PX_LAST only

    #Get candidate Pairs
    
    All_Functions.Get_candidate_pairs(data)
    
    #load candidate Pairs
    
    candidate_pairs_overbought = pd.read_excel("candidate_pairs_overbought.xlsx")
    candidate_pairs_candy_fish = pd.read_excel("candidate_pairs_candy_fish.xlsx")
    candidate_pairs_rolled_over = pd.read_excel("candidate_pairs_rolled_over.xlsx")
    candidate_pairs_crossed_over = pd.read_excel("candidate_pairs_crossed_over.xlsx")

    #Get top 10 pairs
    
    All_Functions.top_10(candidate_pairs_overbought, candidate_pairs_candy_fish, candidate_pairs_rolled_over, candidate_pairs_crossed_over)
    
    #Load top 10 pairs
    
    top10_pairs = (pd.read_csv("top10_list.csv", index_col = 0).iloc[:,0]).tolist()
    
    #Get factors
    
    factors_cleaned = All_Functions.Factor_Cleaning_All(macro_raw, feedstock_raw)
    macro_factors = factors_cleaned[0]
    feedstock_factors = factors_cleaned[1]
    
    #Run Stepwise Regression
    
    All_Functions.stepwise_factors(macro_factors, feedstock_factors, top10_pairs, price_data)
    
    #Load the stepwise regression results
    
    lag_factors = pd.read_csv('stepwise_factors.csv', index_col = 0)

    #Run the Random Forest 
    
    All_Functions.Random_Forest_Code(top10_pairs, price_data, macro_factors, feedstock_factors, lag_factors)
    
    #Run the Logistic Regression
    
    print("Running Logit")
    
    All_Functions.Logistic_Regression_Code(top10_pairs, price_data, macro_factors, feedstock_factors, lag_factors)
    
    #Run the Commodity Random Forest
    
    print("Running Comm")
    
    All_Functions.Commodity_RF_Code(top10_pairs, price_data, macro_factors, feedstock_factors, lag_factors)


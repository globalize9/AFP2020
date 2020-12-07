#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from itertools import combinations
import pymannkendall as mk
from IPython.display import clear_output
import time
from sklearn.linear_model import LinearRegression
import scipy.stats as sp
import pickle



# # Getting Candidate Pairs

# In[ ]:


all_data_2 = pd.read_excel("Final_Data.xlsx", sheet_name = "Sheet3", header = [0,1])
# all_data_2 = pd.read_excel("Data_Full_Editable.xlsx", sheet_name = "Sheet3", header = [0,1])

all_data_2.index = all_data_2['Unnamed: 0_level_0']['Dates']

all_data_2_raw = all_data_2.drop(columns ='Unnamed: 0_level_0' )

all_data_2 = all_data_2_raw[all_data_2_raw.index <= "2019-12-31"] # specify end_date

companies = np.array(all_data_2.columns.get_level_values(0).unique()) #All the unique companies

all_combinations = []
for i in combinations(companies,2):
    all_combinations.append((i[0]+ "_" + i[1]))

all_combinations


# In[3]:


candidate_pairs_overbought = pd.DataFrame({'Return_to_trend':[], 'Trend_adj_peak_trough':[],
                                           'SD_moves':[], 'Num_above_trend':[], 'Num_below_trend':[], 'Num_above_200MA':[], 'Num_below_200MA':[], 'peaking':[], 'troughing':[]})
candidate_pairs_candy_fish = candidate_pairs_overbought.copy()
candidate_pairs_rolled_over = candidate_pairs_overbought.copy()
candidate_pairs_crossed_over = candidate_pairs_overbought.copy()
candidate_pairs_put_call = candidate_pairs_overbought.copy()

def get_RSI_14day(data):
    df_1_new = data.reset_index().copy()
    df_1_new.loc[:,'avg_14up'] = 0
    df_1_new.loc[:,'avg_14down'] = 0
    df_1_new.loc[13,'avg_14up'] = df_1_new.loc[range(0,14),'Up'].mean()
    df_1_new.loc[13,'avg_14down'] = df_1_new.loc[range(0,14),'Down'].mean()
    for i in range(14,df_1_new.shape[0]):
        df_1_new.loc[i,'avg_14up'] = (df_1_new.loc[(i-1),'avg_14up']*13 + df_1_new.loc[i,'Up'])/14
        df_1_new.loc[i,'avg_14down'] = (df_1_new.loc[(i-1),'avg_14down']*13 + df_1_new.loc[i,'Down'])/14
    df_1_new['RS'] = df_1_new['avg_14up']/df_1_new['avg_14down']
    df_1_new['RSI'] = (100 - (100/(1+df_1_new['RS'])))
    return(np.array(df_1_new['RSI']))

all_indicators = pd.DataFrame({'Index':[0],
                          'RSI':[0],
                          'Std.dev_PE':[0],
                          'Avg_PE':[0],
                          'PE':[0],
                          'Recent_Peak':[0],
                          'MoM':[0],
                          '3M_MoM':[0],
                          'YoY':[0],
                          'Max_YoY':[0],
                          '50_MA':[0],
                          '200_MA':[0],
                          '50_day_test_peak':[0],
                          '50_day_test_trough':[0],
                          'Min_YoY':[0],
                          'PC_EWM20':[0]})


# In[4]:


start_trend = '2015-01-01'

def LinearTrend(dataset):
    # input data series with dates on the index, i.e. df_1['Index']
    dataset_subset = dataset.loc[start_trend:]
    y=np.array(dataset_subset.values, dtype=float)
    x=np.array(pd.to_datetime(dataset_subset.index.values), dtype=float)
    slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
    yf = (slope*x)+intercept    
    return pd.DataFrame(yf, columns = ['trend'], index = dataset[start_trend:].index)

def TrendAdjPT(dataset, trend_data):
    last_peak_loc = df_1_test.index[np.where(df_1_test.Peak==1)[0][-1]]
    
    trend_factor = trend_data.iloc[len(trend_data)-1] / trend_data[trend_data.index == last_peak_loc] - 1
    return trend_factor.values

def Days_above_below(dataset, trend_data):    
    df_1_test = pd.DataFrame(dataset).copy()
    df_1_test.loc[:,'Trend'] = trend_data.loc[:,'trend']
    df_1_test.loc[:,'Index_200MA'] = df_1_test.loc[:,'Index'].rolling(window = 200).mean()
    df_1_test['Above_Below'] = df_1_test.apply(lambda x: 'Above' if x['Index']>x['Trend'] else 'Below' if x['Index']<x['Trend'] else np.nan, axis = 1)
    df_1_test['Above_Below_200MA'] = df_1_test.apply(lambda x: 'Above' if x['Index']>x['Index_200MA'] else 'Below' if x['Index']<x['Index_200MA'] else np.nan, axis = 1)
    Num_above_trend = df_1_test[df_1_test['Above_Below']=="Above"].shape[0]
    Num_below_trend = df_1_test[df_1_test['Above_Below']=="Below"].shape[0]
    Num_above_200MA = df_1_test[df_1_test['Above_Below_200MA']=="Above"].shape[0]
    Num_below_200MA = df_1_test[df_1_test['Above_Below_200MA']=="Below"].shape[0]
    
    return Num_above_trend, Num_below_trend, Num_above_200MA, Num_below_200MA

def SDmoves(ind_data, trend_data):
    diff = abs(ind_data - trend_data.iloc[:,0])
    sd_moves = np.std(diff)
    return sd_moves

def Perc_above_below(dataset, trend_data):
    ab_factor = trend_data.iloc[len(trend_data)-1] / dataset.iloc[len(dataset)-1]  - 1
    return ab_factor.values


# In[ ]:



start = dt.datetime.now()
for i in range(len(all_combinations)):
    print(i)
    df_1= pd.DataFrame({'Index':[],
                        'PE_Ratio':[],
                        'Div_Diff':[]})
    
    ###### calculating the technical indicators
    df_1['Index'] = all_data_2[str(all_combinations[i].split('_')[0])]['PX_LAST']/all_data_2[str(all_combinations[i].split('_')[1])]['PX_LAST']
    df_1['PE_Ratio'] = all_data_2[str(all_combinations[i].split('_')[0])]['PE_RATIO']/all_data_2[str(all_combinations[i].split('_')[1])]['PE_RATIO']
    df_1['Div_Diff'] = (all_data_2[str(all_combinations[i].split('_')[0])]['AVERAGE_DIVIDEND_YIELD'] - all_data_2[str(all_combinations[i].split('_')[1])]['AVERAGE_DIVIDEND_YIELD'])

    if df_1['Index'].isna().all(): continue 
    
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
    
    
    df_1 = df_1.dropna(subset = ['Index'])
    df_1['RSI'] = get_RSI_14day(df_1)
    
    
    # start = dt.datetime.strptime('2010-01-01', '%Y-%m-%d')
    end = dt.datetime.strptime('2019-12-31', '%Y-%m-%d')
    df_1_test = df_1.loc[:end,]
    
    df_1_test.loc[:,'20_day_EWM'] = df_1_test['Index'].ewm(span = 20).mean()
    df_1_test.loc[:,'First_Derivative'] = (df_1_test['20_day_EWM'].shift(1) - df_1_test['20_day_EWM'].shift(-1))/2
    df_1_test.loc[:,'Lag_First_Derivative'] = df_1_test['First_Derivative'].shift(1)
    df_1_test.loc[:,'Peak'] = df_1_test.apply(lambda x: 1 if (x['Lag_First_Derivative'] > 0 and x['First_Derivative'] < 0) else 0, axis = 1)
    # df_1_test.loc[:,'PC_Ratio_EWM20'] = df_1_test['Put_Call_Ratio'].ewm(span = 20).mean()
    # 20 day's EWM for PC_Ratio b/c https://www.investopedia.com/trading/forecasting-market-direction-with-put-call-ratios/
    
    indicators = pd.DataFrame({'Index':[0],
                               'RSI':[0],
                              'Std.dev_PE':[0],
                              'Avg_PE':[0],
                              'PE':[0],
                              'Recent_Peak':[0],
                              'MoM':[0],
                              '3M_MoM':[0],
                              'YoY':[0],
                              'Max_YoY':[0],
                              '50_MA':[0],
                              '200_MA':[0],
                              '50_day_test_peak':[0],
                              '50_day_test_trough':[0],
                              'Min_YoY':[0],
                              'PC_EWM20':[0]})
    
    # one year look back period for PE
    end_date = '2019-12-31' # we can adjust this to the last date of the data, use this for backtest 
    end = dt.datetime.strptime(end_date, '%Y-%m-%d') # we can adjust this to the last date of the data, use this for backtest 
    start = end - dt.timedelta(days = 365)
    indicators['Index'] = df_1_test.loc[end,'Index']
    indicators['RSI'] = df_1_test.loc[end,'RSI']
    indicators['Std.dev_PE'] = df_1_test.loc[start:end,'PE_Ratio'].std()
    indicators['Avg_PE'] = df_1_test.loc[start:end,'PE_Ratio'].mean()
    indicators['PE'] = df_1_test.loc[end,'PE_Ratio']
    indicators['Recent_Peak'] = df_1_test[df_1_test.Peak==1].iloc[-1]['Index']
    indicators['MoM'] = df_1_test.loc[end,'MoM_change']
    indicators['3M_MoM'] = df_1_test.loc['2019-10-01':end,'MoM_change'].mean()
    indicators['YoY'] = df_1_test.loc[end,'YoY_change']
    indicators['Max_YoY'] = df_1_test['YoY_change'].rolling(window = 252).max()[-1]
    indicators['Min_YoY'] = df_1_test['YoY_change'].rolling(window = 252).min()[-1]
    indicators['50_MA'] = df_1_test.iloc[-1]['50_MA']
    indicators['200_MA'] = df_1_test.iloc[-1]['200_MA']
    # 1 day shift for the time being, may change to 10 depending on what Laurence prefers
    indicators['50_day_test_peak'] = ((df_1_test['50_MA']- df_1_test['200_MA']).shift(periods = 1).rolling(window = 126).min() > 0) [-1]
    indicators['50_day_test_trough'] = ((df_1_test['50_MA']- df_1_test['200_MA']).shift(periods = 1).rolling(window = 126).max() < 0) [-1]
    # indicators['PC_EWM20'] = df_1_test.iloc[-1]['PC_Ratio_EWM20']
    
    
    check_peak_overbought = ((indicators['RSI'] >70) & (indicators['PE'] > (indicators['Avg_PE']+indicators['Std.dev_PE'])))
    check_peak_candy_fish = ((indicators['Index'] < indicators['Recent_Peak']*0.98) & ((indicators['MoM']> 0) & (indicators['3M_MoM'] > 0)))
    check_peak_rolled_over = ((indicators['YoY'] >=  indicators['Max_YoY']*0.5))
    check_peak_crossed_over = ((indicators['50_MA']< indicators['200_MA']) & indicators['50_day_test_peak'])
    # check_peak_put_call = indicators['PC_EWM20'] >= 2
    
    check_trough_overbought = ((indicators['RSI'] <30) & (indicators['PE'] < (indicators['Avg_PE']-indicators['Std.dev_PE']))) 
    check_trough_candy_fish = ((indicators['Index'] > indicators['Recent_Peak']*1.02) & ((indicators['MoM'] < 0) & (indicators['3M_MoM'] <0))) 
    check_trough_rolled_over = ((indicators['YoY'] <=  indicators['Min_YoY']*0.5)) 
    check_trough_crossed_over = ((indicators['50_MA'] > indicators['200_MA']) & (indicators['50_day_test_trough']))    
    # check_trough_put_call = indicators['PC_EWM20'] <= 1/2
    
    all_indicators = all_indicators.append(indicators, ignore_index = True)
    
    # trend analysis
    def trend_analysis(dataset):
        trend_data = LinearTrend(dataset)   
        trend_indicators = pd.DataFrame({'Return_to_trend':[0], # Sanchita
                                          'Trend_adj_peak_trough':[0], # Yushi
                                          'SD_moves':[0], # Jo
                                          'Num_above_trend':[0], # Shubham
                                          'Num_below_trend':[0],
                                          'Num_above_200MA':[0],
                                          'Num_below_200MA':[0]}) # Shubham
                
        trend_indicators['Trend_adj_peak_trough'] = TrendAdjPT(dataset, trend_data)
        trend_indicators[['Num_above_trend', 'Num_below_trend', 'Num_above_200MA', 'Num_below_200MA']] = Days_above_below(dataset, trend_data)
        trend_indicators['SD_moves'] = SDmoves(ind_data = dataset, trend_data = trend_data)
        trend_indicators['Return_to_trend'] = Perc_above_below(dataset, trend_data)
        return trend_indicators
    
    if (check_peak_overbought[0]== True):
        #candidate_pairs_overbought = candidate_pairs_overbought.append({'peaking':all_combinations[i]}, ignore_index = True)
        temp_trend = trend_analysis(df_1_test['Index'])
        temp_trend['peaking'] = all_combinations[i]
        temp_trend['troughing'] = np.NaN
        candidate_pairs_overbought = candidate_pairs_overbought.append(temp_trend, ignore_index = True)
    elif(check_trough_overbought[0]==True):
        #candidate_pairs_overbought = candidate_pairs_overbought.append({'troughing':all_combinations[i]}, ignore_index = True)
        temp_trend = trend_analysis(df_1_test['Index'])
        temp_trend['peaking'] = np.NaN
        temp_trend['troughing'] = all_combinations[i] 
        candidate_pairs_overbought = candidate_pairs_overbought.append(temp_trend, ignore_index = True)
    
    if (check_peak_candy_fish[0]== True):
        #candidate_pairs_candy_fish = candidate_pairs_candy_fish.append({'peaking':all_combinations[i]}, ignore_index = True)
        temp_trend = trend_analysis(df_1_test['Index'])
        temp_trend['peaking'] = all_combinations[i]
        temp_trend['troughing'] = np.NaN
        candidate_pairs_candy_fish = candidate_pairs_candy_fish.append(temp_trend, ignore_index = True)
    elif(check_trough_candy_fish[0]==True):
        #candidate_pairs_candy_fish = candidate_pairs_candy_fish.append({'troughing':all_combinations[i]}, ignore_index = True)
        temp_trend = trend_analysis(df_1_test['Index'])
        temp_trend['peaking'] = np.NaN
        temp_trend['troughing'] = all_combinations[i]
        candidate_pairs_candy_fish = candidate_pairs_candy_fish.append(temp_trend, ignore_index = True)
        
    if (check_peak_rolled_over[0]== True):
        #candidate_pairs_rolled_over = candidate_pairs_rolled_over.append({'peaking':all_combinations[i]}, ignore_index = True)
        temp_trend = trend_analysis(df_1_test['Index'])
        temp_trend['peaking'] = all_combinations[i]
        temp_trend['troughing'] = np.NaN
        candidate_pairs_rolled_over = candidate_pairs_rolled_over.append(temp_trend, ignore_index = True)
    elif(check_trough_rolled_over[0]==True):
        #candidate_pairs_rolled_over = candidate_pairs_rolled_over.append({'troughing':all_combinations[i]}, ignore_index = True)
        temp_trend = trend_analysis(df_1_test['Index'])
        temp_trend['peaking'] =  np.NaN
        temp_trend['troughing'] = all_combinations[i]
        candidate_pairs_rolled_over = candidate_pairs_rolled_over.append(temp_trend, ignore_index = True)
        
    if (check_peak_crossed_over[0]== True):
        #candidate_pairs_crossed_over = candidate_pairs_crossed_over.append({'peaking':all_combinations[i]}, ignore_index = True)
        temp_trend = trend_analysis(df_1_test['Index'])
        temp_trend['peaking'] = all_combinations[i] 
        temp_trend['troughing'] = np.NaN
        candidate_pairs_crossed_over = candidate_pairs_crossed_over.append(temp_trend, ignore_index = True)
    elif(check_trough_crossed_over[0]==True):
        #candidate_pairs_crossed_over = candidate_pairs_crossed_over.append({'troughing':all_combinations[i]}, ignore_index = True)
        temp_trend = trend_analysis(df_1_test['Index'])
        temp_trend['peaking'] =  np.NaN
        temp_trend['troughing'] = all_combinations[i]
        candidate_pairs_crossed_over = candidate_pairs_crossed_over.append(temp_trend, ignore_index = True)
    
# =============================================================================
#     # not running p/c for now since there are a lot of pairs here
#     if (check_peak_put_call[0]== True):
#         temp_trend = trend_analysis(df_1['Index'])
#         temp_trend['peaking'] = all_combinations[i] 
#         temp_trend['troughing'] = np.NaN
#         candidate_pairs_put_call = candidate_pairs_put_call.append(temp_trend, ignore_index = True)
#     elif(check_trough_put_call[0]==True):
#         # candidate_pairs_put_call = candidate_pairs_put_call.append({'troughing':all_combinations[i]}, ignore_index = True)
#         temp_trend = trend_analysis(df_1['Index'])
#         temp_trend['peaking'] = np.NaN 
#         temp_trend['troughing'] = all_combinations[i]
#         candidate_pairs_put_call = candidate_pairs_put_call.append(temp_trend, ignore_index = True)
# =============================================================================
        
    
    clear_output(wait = True)


# In[6] saving the candidate pairs:


candidate_pairs_overbought.to_excel("candidate_pairs_overbought.xlsx")
candidate_pairs_candy_fish.to_excel("candidate_pairs_candy_fish.xlsx")
candidate_pairs_rolled_over.to_excel("candidate_pairs_rolled_over.xlsx")
candidate_pairs_crossed_over.to_excel("candidate_pairs_crossed_over.xlsx")
# candidate_pairs_put_call.to_excel('candidate_pairs_put_call.xlsx')

all_indicators = all_indicators.loc[1:len(all_indicators)]
all_indicators.index = all_combinations

all_indicators.to_excel('all_indicators.xlsx')


# In[4] # Loading Selected Candidated Pairs:


candidate_pairs_overbought = pd.read_excel("candidate_pairs_overbought.xlsx")
candidate_pairs_candy_fish = pd.read_excel("candidate_pairs_candy_fish.xlsx")
candidate_pairs_rolled_over = pd.read_excel("candidate_pairs_rolled_over.xlsx")
candidate_pairs_crossed_over = pd.read_excel("candidate_pairs_crossed_over.xlsx")


# ## Ranking Pairs and Selecting Top 10 in each

# In[26]:


for i in [candidate_pairs_overbought, candidate_pairs_candy_fish, candidate_pairs_rolled_over, candidate_pairs_crossed_over]:
    i['Rank_Return_to_trend'] = abs(i['Return_to_trend'].rank())
    i['Rank_Trend_adj_peak_trough'] = abs(i['Trend_adj_peak_trough'].rank())
    i['Rank_SD_moves'] = i['SD_moves'].rank()
    i['Rank_Num_trend'] = np.where(i['peaking'].isna(), i['Num_below_trend'], i['Num_above_trend'])
    i['Rank_Num_trend'] = i['Rank_Num_trend'].rank()
    i['Rank_Num_200MA'] = np.where(i['peaking'].isna(), i['Num_below_200MA'], i['Num_above_200MA'])
    i['Rank_Num_200MA'] = i['Rank_Num_200MA'].rank()
    i['avg_rank'] = (i['Rank_Return_to_trend'] + i['Rank_Trend_adj_peak_trough'] + i['Rank_SD_moves'] + i['Rank_Num_trend'] + i['Rank_Num_200MA'])/5
    i['Pair'] = np.where(i['peaking'].isna(), i['troughing'], i['peaking'])


# In[78]:


top10 = np.array([])
for i in [candidate_pairs_overbought, candidate_pairs_candy_fish, candidate_pairs_rolled_over, candidate_pairs_crossed_over]:
    top10 = np.append(top10, i.sort_values(by = ['avg_rank'])['Pair'][0:10].values)

top10_pairs = top10.tolist()

# In[4]:


all_data_2 = pd.read_excel("Final_Data.xlsx", sheet_name = "Sheet3", header = [0,1])
# all_data_2 = pd.read_excel("Data_Full_Editable.xlsx", sheet_name = "Sheet3", header = [0,1])

all_data_2.index = all_data_2['Unnamed: 0_level_0']['Dates']

all_data_2_raw = all_data_2.drop(columns ='Unnamed: 0_level_0' )

all_data_2 = all_data_2_raw[all_data_2_raw.index <= "2019-12-31"] # specify end_date

price_data = all_data_2.xs('PX_LAST', axis = 1, level = 1, drop_level = False) # subsetting to PX_LAST only


# # Clean Factors

# In[5]:

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

macro_factors = YoYClean(macro_factors_round1) / 100
feedstock_factors = YoYClean(feedstock_factors_round1) 
feedstock_factors = (feedstock_factors_round1 / feedstock_factors_round1.shift(252) - 1) * 100
feedstock_factors = feedstock_factors.dropna(axis = 0) / 100



# # Stepwise Regression

# In[382]:

price_data = all_data_2.xs('PX_LAST', axis = 1, level = 1, drop_level = False) # subsetting to PX_LAST only
price_data.columns = price_data.columns.droplevel(level = 1)
#price_data = price_data.loc['2015-01-01':]
#price_data = price_data.dropna(axis = 1)


# In[383]:


stepwise_factors = pd.DataFrame({'pair':top10_pairs,
             'factor_1':[np.nan]*len(top10_pairs),
             'lag_1':[np.nan]*len(top10_pairs),
             'factor_2':[np.nan]*len(top10_pairs),
             'lag_2':[np.nan]*len(top10_pairs),
             'factor_3':[np.nan]*len(top10_pairs),
             'lag_3':[np.nan]*len(top10_pairs),
             'factor_4':[np.nan]*len(top10_pairs),
             'lag_4':[np.nan]*len(top10_pairs),
             'factor_5':[np.nan]*len(top10_pairs),
             'lag_5':[np.nan]*len(top10_pairs)})


# In[384]:


def R_2_stepwise(main_data, factor_data, lag):
    main_data_w_factor = pd.merge(main_data, factor, left_index = True, right_index = True, how = 'left')
    main_data_w_factor['lag_factor'] = main_data_w_factor.iloc[:,1].shift(22*lag)
    main_data_w_factor = main_data_w_factor.dropna()
    if (main_data_w_factor.shape[0]>0):
        lm = LinearRegression()
        lm.fit(main_data_w_factor[['lag_factor']], main_data_w_factor[['Index']],)
        R_2 = np.round(lm.score(main_data_w_factor[['lag_factor']], main_data_w_factor[['Index']]),5)
    else:
        R_2 = 0
    return(R_2)


# In[385]:


all_factors = pd.merge(macro_factors, feedstock_factors, left_on = macro_factors.index, right_on = feedstock_factors.index, how = 'outer')
all_factors.index = all_factors.key_0
all_factors.index.name = 'Date'
all_factors = all_factors.drop(columns = ['key_0'])


# In[ ]:


for pair in range(len(top10_pairs)):
    print(pair)
        
    ################ Get all Highest R_2 #######################
    
    current_factor_level = 1
    
    R_2_df = pd.DataFrame({'factor_name':[],
                       'lag':[],
                      'R_2':[]})
    
    highest_R_2_factor = np.array([])
    lag_of_highest_R_2 = np.array([])
    
    # insert code to read the candidate pairs list
    pair_names = top10_pairs[pair]
    pair_names = pair_names.split('_')
    
    main_data = price_data.loc[:,price_data.columns.isin(pair_names)].copy().dropna()
    
    if main_data.shape[1] < 2:
        continue
    
    main_data['Index'] = (main_data.iloc[:,0]/main_data.iloc[:,1])
    main_data[['Index']] = (main_data[['Index']].shift(-21*6) - main_data[['Index']])/main_data[['Index']]
    factors_to_consider = all_factors.columns
    
    for j in factors_to_consider:
        for lag in range(1,19,1):
            factor = all_factors.loc[:,[j]]
            R_2 = R_2_stepwise(main_data[['Index']], factor, lag)
            R_2_df = R_2_df.append({'factor_name':j,
                          'lag':lag,
                          'R_2':R_2}, ignore_index = True)
    
    highest_R_2_factor = np.append(highest_R_2_factor, np.array(R_2_df.loc[R_2_df.R_2 == max(R_2_df.R_2),'factor_name']))
    lag_of_highest_R_2 = np.append(lag_of_highest_R_2, int(np.array(R_2_df.loc[R_2_df.R_2 == max(R_2_df.R_2),'lag'])[0]))
    
    stepwise_factors.loc[stepwise_factors.pair == top10_pairs[pair],'factor_1'] = highest_R_2_factor[0]
    stepwise_factors.loc[stepwise_factors.pair == top10_pairs[pair],'lag_1'] = lag_of_highest_R_2[0]
    
    
    ####################### Step 2 in Stepwise Regression #####################
    
    for i in range(4):
        
        current_factor_level = i+2

        factors_to_consider = factors_to_consider[~factors_to_consider.isin(highest_R_2_factor)]    
        ## We need the error with highest R_2 factor then run with other remaining factors

        ##### Extract errors from first factor regression #####

        reg_data = pd.merge(main_data[['Index']], all_factors.loc[:,highest_R_2_factor[0:(i+1)]], left_index = True, right_index = True, how = 'left')
        for main_lag in range(i+1):
            reg_data[['lag_factor_' + str(main_lag)]] = reg_data.loc[:,[highest_R_2_factor[main_lag]]].shift(22*int(lag_of_highest_R_2[main_lag-1]))
        reg_data = reg_data.dropna()
        lm = LinearRegression()
        lm.fit(reg_data[['lag_factor_' + str(k) for k in range(i+1)]], reg_data[['Index']])
        errors = lm.predict(reg_data[['lag_factor_' + str(k) for k in range(i+1)]]) - reg_data[['Index']]
        errors.columns = ['Index']

        R_2_df = pd.DataFrame({'factor_name':[],
                           'lag':[],
                          'R_2':[]})

        for j in factors_to_consider:
            for lag in range(1,19,1):
                factor = all_factors.loc[:,[j]]
                R_2 = R_2_stepwise(errors, factor, lag)
                R_2_df = R_2_df.append({'factor_name':j,
                              'lag':lag,
                              'R_2':R_2}, ignore_index = True)

        highest_R_2_factor = np.append(highest_R_2_factor,np.array(R_2_df.loc[R_2_df.R_2 == max(R_2_df.R_2),'factor_name']))
        lag_of_highest_R_2 = np.append(lag_of_highest_R_2, int(np.array(R_2_df.loc[R_2_df.R_2 == max(R_2_df.R_2),'lag'])[0]))

        stepwise_factors.loc[stepwise_factors.pair == top10_pairs[pair],f'factor_{i+2}'] = highest_R_2_factor[i+1]
        stepwise_factors.loc[stepwise_factors.pair == top10_pairs[pair],f'lag_{i+2}'] = lag_of_highest_R_2[i+1]      

        factors_to_consider = factors_to_consider[~factors_to_consider.isin(highest_R_2_factor)]   
            
            


# In[391] exporting stepwise factors:


stepwise_factors.to_csv("stepwise_factors.csv")


# In[ ]:
# lag factors implementing
lag_factors = pd.read_csv(r'C:\Users\yushi\Documents\GitHub\AFP2020\stepwise_factors.csv')
lag_factors.columns

def CheckDate(date_in,this_list):
    while date_in not in this_list:
        date_in -= dt.timedelta(days = 1)
    return date_in


prediction_direction = dict.fromkeys(top10_pairs)

# reading in raw data, adapted from data_cleaning
all_data_2 = pd.read_excel("data_0719.xlsx", sheet_name = "Sheet5", header = [0,1])
all_data_2.index = all_data_2['Unnamed: 0_level_0']['Dates']
all_data_2_raw = all_data_2.drop(columns ='Unnamed: 0_level_0' )
price_data = all_data_2.xs('PX_LAST', axis = 1, level = 1, drop_level = False) # subsetting to PX_LAST only

def PlotIndex(i):
    pair_names = top10_pairs[i]
    pair_names = pair_names.split('_')
    
    # calculating the index as the ratio of PX_LAST
    index_level = price_data[pair_names[0]] / price_data[pair_names[1]]
    index_level = index_level.dropna()
    
    index_level.plot(title = top10_pairs[i])

for i in [x for x in range(len(top10_pairs)) if x != 3]:
    PlotIndex(-1)

# could also add a portion for back test...i.e. calculate the actual results vs. predicted
    

for i in [x for x in range(len(top10_pairs)) if x != 3]:
    # insert code to read the candidate pairs list
    pair_names = top10_pairs[i]
    pair_names = pair_names.split('_')
    
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

    # need a day input, [last day of the training dataset]
    # puting in an arbitrary date for now
    input_date = dt.date(2019,12,30)
    
    nameP = top10_pairs[i]
    row_num = np.where(nameP == lag_factors.pair)[0][0]
    lag_factor_names = lag_factors.loc[row_num,['factor_1','factor_2','factor_3','factor_4','factor_5']]
    lag_factor_times = lag_factors.loc[row_num,['lag_1','lag_2','lag_3','lag_4','lag_5']]
    days_lag = lag_factor_times * 21 # converting to days equivalent
    spec_dates = [input_date - dt.timedelta(days = int(x)) for x in days_lag.tolist()]
    lag_dates = spec_dates
    
    
    
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
    
    ################################ RF Implementation ################################
    
    # labels are the values we want to predict
    labels = np.array(features['PX_LAST'])
    
    # Remove the labels from the features
    features = features.drop('PX_LAST', axis = 1)
    
    # Saving feature names for later use
    feature_list = list(features.columns)
    
    # Convert to numpy array
    features = np.array(features)
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state = 42)
    from sklearn.model_selection import RandomizedSearchCV
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 25, stop = 150, num = 15)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(3, 15, num = 2)]
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
    start_time = dt.datetime.now()
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
    
    # the random search picks different combinations, whereas the grid search is more refinement but slower
    end = dt.datetime.now()
    print(end - start_time)
    
    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
    
    from sklearn.model_selection import GridSearchCV
    # Create the parameter grid based on the results of random search    
    def GridAdjust(rf_best_params):
        temp_grid = rf_best_params.copy()
        p_grid = temp_grid.copy()
        p_grid['max_depth'] = [temp_grid['max_depth']-1, temp_grid['max_depth']+1]
        p_grid['min_samples_leaf'] = [max(temp_grid['min_samples_leaf']-5,1), temp_grid['min_samples_leaf']+5]
        p_grid['min_samples_split'] = [max(temp_grid['min_samples_split']-10,1), temp_grid['min_samples_split']+10]
        p_grid['n_estimators'] = [max(temp_grid['n_estimators']-15,1), temp_grid['n_estimators']+15]
        p_grid['max_features'] = [temp_grid['max_features']]
        p_grid['bootstrap'] = [temp_grid['bootstrap']]
        return p_grid
    
    param_grid = GridAdjust(rf_random.best_params_)
    
    # Create a base model
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
        RMSE = np.sqrt(np.mean(np.square(abs(predictions - test_labels))))
        print('Model Performance')
        print('Average Error: {:0.4f}'.format(np.mean(errors)))
        print('RMS Error: {:0.4f}'.format(RMSE))
        return mape
    
    grid_test_mape = evaluate(best_grid, test_features, test_labels)
    grid_train_mape = evaluate(best_grid, train_features, train_labels)
    
    predictions_final = best_grid.predict(np.array(last_day_X).reshape(1,5))
    prediction_direction[top10_pairs[i]] = np.sign(predictions_final[0])
    # +'ve, therefore peaking
    
    ## logistics is % change instead.... (6 month ahead - today)/today's level
    ## check to see if it is greater than 15% threshold 

# pair 3 is blank...filling it in for now
    
# saving pickle 
filename = 'prediction_direction'
outfile = open(filename, 'wb')
pickle.dump(prediction_direction, outfile)
outfile.close()

# opening pickle
infile = open('prediction_direction','rb')
new_dict = pickle.load(infile)
infile.close()



##########Logistic Regression################
lag_factors = pd.read_csv('stepwise_factors.csv') #TO CHANGE

prediction_proba = dict.fromkeys(top10_pairs)

for i in range(len(prediction_proba)):
    print(i)
    if i == 5:
        continue
    else:
        
        pair_name = top10_pairs[i]
        print(pair_name)
        pair_names = pair_name.split('_')

        index_level = price_data[pair_names[0]] / price_data[pair_names[1]]
        index_level = index_level.loc[macro_factors.index] # aligns the y data with the x data
        index_level = index_level.dropna()

        temp_macro = macro_factors.loc[index_level.index]
        temp_feedstock = feedstock_factors.loc[index_level.index]

        temp_macro = temp_macro.fillna(0)
        temp_feedstock = temp_feedstock.fillna(0)

        nameP = top10_pairs[i]
        row_num = np.where(nameP == lag_factors.pair)[0][0]
        lag_factor_names = lag_factors.loc[row_num,['factor_1','factor_2','factor_3','factor_4','factor_5']]
        lag_factor_times = lag_factors.loc[row_num,['lag_1','lag_2','lag_3','lag_4','lag_5']]
        days_lag = lag_factor_times * 21 # converting to days equivalent
        # spec_dates = [input_date - dt.timedelta(days = int(x)) for x in days_lag.tolist()]
        # lag_dates = spec_dates


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

        Y_var = (index_level.shift(periods = -21*6) - index_level)/index_level # 6 months forward

        joint_df = X_vars.copy()
        joint_df['PX_LAST'] = Y_var[X_vars.index]
        last_day_X = joint_df.iloc[-1,:].drop('PX_LAST', axis = 0)
        joint_df = joint_df.dropna()
        features = joint_df.copy()

        joint_df['Direction'] = [max(0,x) for x in np.sign(joint_df['PX_LAST'])]

        logit = LogisticRegression()

        logit.fit(joint_df.iloc[:,~joint_df.columns.isin(["Direction","PX_LAST"])], np.array(joint_df.loc[:,["Direction"]]).ravel())

        proba = logit.predict_proba(pd.DataFrame(last_day_X).transpose())[0][1]

        prediction_proba[pair_name] = np.round(proba,4)


filename = 'prediction_probability'
outfile = open(filename, 'wb')
pickle.dump(prediction_proba, outfile)
outfile.close()

# opening pickle
infile = open('prediction_probability','rb')
prediction_probability = pickle.load(infile)
infile.close()


## pick the top 5 pairs based on logistic


####### commodities regression 
# using retail gasoline price in-lieu of Gasoline (RBOB) due to incomplete data 
# that could be remedied by not screening as intensively in the previous macro dataset screen
commodities_X = feedstock_factors[['Corn','Cotton','Natural Gas','Oil (Brent)','Retail Gasoline Price','Soy','Sugar','Wheat']]
commodities_X_lag6 = commodities_X.shift(periods = 6*21)
commodities = lag_factors.copy()
commodities = commodities.drop('Unnamed: 0', axis = 1)


for i in range(len(commodities)):
    # insert code to read the stepwise pairs list
    pair_names = commodities.loc[i,'pair']
    pair_names = pair_names.split('_')
    
    # calculating the index as the ratio of PX_LAST
    index_level = price_data[pair_names[0]] / price_data[pair_names[1]]
    index_level = index_level.loc[macro_factors.index] # aligns the y data with the x data
    index_level = index_level.dropna()

    # need a day input, [last day of the training dataset]
    # puting in an arbitrary date for now
    input_date = dt.date(2019,12,30)
    
    nameP = commodities.loc[i,'pair']
    row_num = np.where(nameP == lag_factors.pair)[0][0]
    lag_factor_names = lag_factors.loc[row_num,['factor_1','factor_2','factor_3','factor_4','factor_5']]
    lag_factor_times = lag_factors.loc[row_num,['lag_1','lag_2','lag_3','lag_4','lag_5']]
                
            
    X_vars = commodities_X_lag6.copy()
    X_vars = X_vars.dropna(axis = 0)
    
    def LagFactorData(factor_name, macro_factors, feedstock_factors):
        if factor_name in macro_factors.columns:
            temp = macro_factors.loc[:,factor_name].copy()
            # temp = temp.shift(int(days_lag[z])) # takes care of the lag
        else:
            temp = feedstock_factors.loc[:,factor_name].copy()
            # temp = temp.shift(int(days_lag[z])) # takes care of the lag
        temp = temp.rename('PX_LAST')
        return temp
        
    
    # running it on the actual stepwise factor
    for z in range(5):
        Y_var = LagFactorData(lag_factor_names[z], macro_factors, feedstock_factors)
        
        # Y_var = index_level.copy()# .shift(periods = -21*6) - index_level # 6 months forward
        
        # matching the dates
        joint_df = X_vars.merge(Y_var, how = 'outer', left_index = True, right_index = True)
        last_day_X = joint_df.iloc[-1,:].drop('PX_LAST', axis = 0)
        joint_df = joint_df.dropna()
        features = joint_df.copy()
        
        ################################ RF Implementation ################################
        
        # labels are the values we want to predict
        labels = np.array(features['PX_LAST'])
        
        # Remove the labels from the features
        features = features.drop('PX_LAST', axis = 1)
        
        # Saving feature names for later use
        feature_list = list(features.columns)
        
        # Convert to numpy array
        features = np.array(features)
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(random_state = 42)
        from sklearn.model_selection import RandomizedSearchCV
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 25, stop = 150, num = 15)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(3, 15, num = 2)]
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
        start_time = dt.datetime.now()
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
        
        # the random search picks different combinations, whereas the grid search is more refinement but slower
        end = dt.datetime.now()
        print(end - start_time)

        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)
        
        # Create the parameter grid based on the results of random search    
        def GridAdjust(rf_best_params):
            temp_grid = rf_best_params.copy()
            p_grid = temp_grid.copy()
            p_grid['max_depth'] = [temp_grid['max_depth']-1, temp_grid['max_depth']+1]
            p_grid['min_samples_leaf'] = [max(temp_grid['min_samples_leaf']-5,1), temp_grid['min_samples_leaf']+5]
            p_grid['min_samples_split'] = [max(temp_grid['min_samples_split']-10,1), temp_grid['min_samples_split']+10]
            p_grid['n_estimators'] = [max(temp_grid['n_estimators']-15,1), temp_grid['n_estimators']+15]
            p_grid['max_features'] = [temp_grid['max_features']]
            p_grid['bootstrap'] = [temp_grid['bootstrap']]
            return p_grid
        
        param_grid = GridAdjust(rf_random.best_params_)
        
        # Create a base model
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
            RMSE = np.sqrt(np.mean(np.square(abs(predictions - test_labels))))
            print('Model Performance')
            print('Average Error: {:0.4f}'.format(np.mean(errors)))
            print('RMS Error: {:0.4f}'.format(RMSE))
            return RMSE
        
        grid_test_rmse = evaluate(best_grid, test_features, test_labels)
        grid_train_rmse = evaluate(best_grid, train_features, train_labels)
        
        # saving the test_rmse into the dataframe
        commodities.loc[i, lag_factor_times.index[z]] = grid_test_rmse

# saving pickle 
filename = 'commodities_rmse'
outfile = open(filename, 'wb')
pickle.dump(commodities, outfile)
outfile.close()


# opening pickle
infile = open('commodities_rmse','rb')
commodities = pickle.load(infile)
infile.close()



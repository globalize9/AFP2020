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
#   df_1_test.loc[:,'PC_Ratio_EWM20'] = df_1_test['Put_Call_Ratio'].ewm(span = 20).mean()
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

end = dt.datetime.now()
end - start


# In[6]:


candidate_pairs_overbought.to_excel("candidate_pairs_overbought.xlsx")
candidate_pairs_candy_fish.to_excel("candidate_pairs_candy_fish.xlsx")
candidate_pairs_rolled_over.to_excel("candidate_pairs_rolled_over.xlsx")
candidate_pairs_crossed_over.to_excel("candidate_pairs_crossed_over.xlsx")
# candidate_pairs_put_call.to_excel('candidate_pairs_put_call.xlsx')

all_indicators = all_indicators.loc[1:len(all_indicators)]
all_indicators.index = all_combinations

all_indicators.to_excel('all_indicators.xlsx')


# # Loading Selected Candidated Pairs

# In[4]:


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


# In[4]:


all_data_2 = pd.read_excel("Final_Data.xlsx", sheet_name = "Sheet3", header = [0,1])
# all_data_2 = pd.read_excel("Data_Full_Editable.xlsx", sheet_name = "Sheet3", header = [0,1])

all_data_2.index = all_data_2['Unnamed: 0_level_0']['Dates']

all_data_2_raw = all_data_2.drop(columns ='Unnamed: 0_level_0' )

all_data_2 = all_data_2_raw[all_data_2_raw.index <= "2019-12-31"] # specify end_date

price_data = all_data_2.xs('PX_LAST', axis = 1, level = 1, drop_level = False) # subsetting to PX_LAST only


# # Clean Factors

# In[5]:


import pandas as pd
import numpy as np
import os


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


# # Stepwise Regression

# In[381]:


top10_pairs = ['AXTA US Equity_HXL US Equity','AVD US Equity_PRLB US Equity','AVD US Equity_TSE US Equity','AVD US Equity_DOW US Equity',
               'AVD US Equity_GCP US Equity','AVD US Equity_VVV US Equity','AVD US Equity_IMCD NA Equity','ADM US Equity_HXL US Equity',
               'AVD US Equity_WDFC US Equity','AVD US Equity_WLK US Equity','CINR US Equity_CE US Equity','GCP US Equity_OEC US Equity',
               'RPM US Equity_UNVR US Equity','ALB US Equity_CC US Equity','CRDA LN Equity_IMCD NA Equity','DSM NA Equity_OEC US Equity',
               'NZYMB DC Equity_OEC US Equity','CINR US Equity_OLN US Equity','KOP US Equity_MMM US Equity','MMM US Equity_OEC US Equity',
               'AXTA US Equity_KWR US Equity','GPRE US Equity_KWR US Equity','AXTA US Equity_DD US Equity','AXTA US Equity_MMM US Equity',
               'AXTA US Equity_HXL US Equity','GPRE US Equity_TSE US Equity','GPRE US Equity_PRLB US Equity','DCI US Equity_KWR US Equity',
               'KOP US Equity_TSE US Equity','AVD US Equity_TSE US Equity','AVD US Equity_BAS GR Equity','AXTA US Equity_NZYMB DC Equity',
               'AVD US Equity_CBT US Equity','GPRE US Equity_RPM US Equity','AVD US Equity_ADM US Equity']


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
            
            


# In[391]:


stepwise_factors.to_csv("stepwise_factors.csv")


# In[ ]:





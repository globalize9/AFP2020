#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from itertools import combinations
import pymannkendall as mk
from IPython.display import clear_output


# In[7]:
# set cwd
os.chdir("C:/Users/yushi/Documents/GitHub/AFP2020")

all_data_2 = pd.read_excel("data_0719.xlsx", sheet_name = "Sheet5", header = [0,1])

all_data_2.index = all_data_2['Unnamed: 0_level_0']['Dates']

all_data_2 = all_data_2.drop(columns ='Unnamed: 0_level_0' )

companies = np.array(all_data_2.columns.get_level_values(0).unique()) #All the unique companies

all_combinations = []
for i in combinations(companies,2):
    all_combinations.append((i[0]+ "_" + i[1]))

all_combinations


# In[149]:


candidate_pairs_overbought = pd.DataFrame({'peaking':[], 'troughing':[]})
candidate_pairs_candy_fish = pd.DataFrame({'peaking':[], 'troughing':[]})
candidate_pairs_rolled_over = pd.DataFrame({'peaking':[], 'troughing':[]})
candidate_pairs_crossed_over = pd.DataFrame({'peaking':[], 'troughing':[]})
candidate_pairs_put_call = pd.DataFrame({'peaking':[], 'troughing':[]})

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


# In[150]:


# for i in range(len(all_combinations)):
for i in range(50):
    print(i)
    df_1= pd.DataFrame({'Index':[],
                        'PE_Ratio':[],
                        'Div_Diff':[]})
    
    ###### calculating the technical indicators
    df_1['Index'] = all_data_2[str(all_combinations[i].split('_')[0])]['PX_LAST']/all_data_2[str(all_combinations[i].split('_')[1])]['PX_LAST']
    df_1['PE_Ratio'] = all_data_2[str(all_combinations[i].split('_')[0])]['PE_RATIO']/all_data_2[str(all_combinations[i].split('_')[1])]['PE_RATIO']
    df_1['Div_Diff'] = (all_data_2[str(all_combinations[i].split('_')[0])]['AVERAGE_DIVIDEND_YIELD'] - all_data_2[str(all_combinations[i].split('_')[1])]['AVERAGE_DIVIDEND_YIELD'])
    df_1['Put_Call_Ratio'] = all_data_2[str(all_combinations[i].split('_')[0])]['PUT_CALL_VOLUME_RATIO_CUR_DAY']/all_data_2[str(all_combinations[i].split('_')[1])]['PUT_CALL_VOLUME_RATIO_CUR_DAY']
    
    # if df_1['Index'].isna().all(): continue 
    
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
    df_1_test.loc[:,'PC_Ratio_EWM20'] = df_1_test['Put_Call_Ratio'].ewm(span = 20).mean()
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
    indicators['PC_EWM20'] = df_1_test.iloc[-1]['PC_Ratio_EWM20']
    
    
    check_peak_overbought = ((indicators['RSI'] >70) & (indicators['PE'] > (indicators['Avg_PE']+indicators['Std.dev_PE'])))
    check_peak_candy_fish = ((indicators['Index'] < indicators['Recent_Peak']*0.98) & ((indicators['MoM']> 0) & (indicators['3M_MoM'] > 0)))
    check_peak_rolled_over = ((indicators['YoY'] >=  indicators['Max_YoY']*0.5))
    check_peak_crossed_over = ((indicators['50_MA']< indicators['200_MA']) & indicators['50_day_test_peak'])
    check_peak_put_call = indicators['PC_EWM20'] >= 2
    
    check_trough_overbought = ((indicators['RSI'] <30) & (indicators['PE'] < (indicators['Avg_PE']-indicators['Std.dev_PE']))) 
    check_trough_candy_fish = ((indicators['Index'] > indicators['Recent_Peak']*1.02) & ((indicators['MoM'] < 0) & (indicators['3M_MoM'] <0))) 
    check_trough_rolled_over = ((indicators['YoY'] <=  indicators['Min_YoY']*0.5)) 
    check_trough_crossed_over = ((indicators['50_MA'] > indicators['200_MA']) & (indicators['50_day_test_trough']))    
    check_trough_put_call = indicators['PC_EWM20'] <= 1/2
    
    all_indicators = all_indicators.append(indicators, ignore_index = True)
    
    if (check_peak_overbought[0]== True):
        candidate_pairs_overbought = candidate_pairs_overbought.append({'peaking':all_combinations[i]}, ignore_index = True)
    elif(check_trough_overbought[0]==True):
        candidate_pairs_overbought = candidate_pairs_overbought.append({'troughing':all_combinations[i]}, ignore_index = True)
    
    if (check_peak_candy_fish[0]== True):
        candidate_pairs_candy_fish = candidate_pairs_candy_fish.append({'peaking':all_combinations[i]}, ignore_index = True)
    elif(check_trough_candy_fish[0]==True):
        candidate_pairs_candy_fish = candidate_pairs_candy_fish.append({'troughing':all_combinations[i]}, ignore_index = True)
        
    if (check_peak_rolled_over[0]== True):
        candidate_pairs_rolled_over = candidate_pairs_rolled_over.append({'peaking':all_combinations[i]}, ignore_index = True)
    elif(check_trough_rolled_over[0]==True):
        candidate_pairs_rolled_over = candidate_pairs_rolled_over.append({'troughing':all_combinations[i]}, ignore_index = True)
        
    if (check_peak_crossed_over[0]== True):
        candidate_pairs_crossed_over = candidate_pairs_crossed_over.append({'peaking':all_combinations[i]}, ignore_index = True)
    elif(check_trough_crossed_over[0]==True):
        candidate_pairs_crossed_over = candidate_pairs_crossed_over.append({'troughing':all_combinations[i]}, ignore_index = True)
    
    if (check_peak_put_call[0]== True):
        candidate_pairs_put_call = candidate_pairs_put_call.append({'peaking':all_combinations[i]}, ignore_index = True)
    elif(check_trough_put_call[0]==True):
        candidate_pairs_put_call = candidate_pairs_put_call.append({'troughing':all_combinations[i]}, ignore_index = True)
            
    
    clear_output(wait = True)

candidate_pairs_overbought.to_excel("candidate_pairs_overbought.xlsx")
candidate_pairs_candy_fish.to_excel("candidate_pairs_candy_fish.xlsx")
candidate_pairs_rolled_over.to_excel("candidate_pairs_rolled_over.xlsx")
candidate_pairs_crossed_over.to_excel("candidate_pairs_crossed_over.xlsx")
candidate_pairs_put_call.to_excel('candidate_pairs_put_call.xlsx')

all_indicators = all_indicators.loc[1:len(all_indicators)]
all_indicators.index = all_combinations


# In[164]:


all_indicators.to_excel('all_indicators.xlsx')


# In[ ]:


### test plots
fig = plt.figure(facecolor='white', figsize = (20,20))
ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
ax0.plot(df_1_test.loc[start:end,'Index'], color = 'blue')
ax0.plot(df_1_test.loc[start:end,'20_day_EWM'])
ax0.grid(True, color='black', linewidth = 0.2)


# In[ ]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





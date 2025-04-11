import os
import sys
import time
import joblib,re
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import traceback
import logging
import warnings
warnings.filterwarnings('ignore')

import tushare as ts
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import interp1d
        
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from xtquant import xtdata

class OPTION:
    def __init__(self):
        pass

    # 获取无风险利率，用SHIBOR
    @staticmethod
    def get_rf(trade_date,pro):
        shibor = pro.shibor(start_date=trade_date, end_date=trade_date)
        rf = pd.DataFrame()
        rf['tau'] = [1,7,14,30,180,270,365]
        rf['rate'] = shibor.iloc[0,1:].reset_index(drop=True)
        interp = interp1d(rf['tau'],rf['rate'],kind="linear",fill_value="extrapolate")
        return interp   
    
    
    # 获取期权数据，并分成近月和次近月两份
    @staticmethod
    def opt_data(trade_date,pro,opt_basic,opt_code='OP510500.SH'):
        opt_price = pro.opt_daily(trade_date=trade_date,exchange="SSE")
        opt_price_temp = opt_price[opt_price['exchange'] == 'SSE'].reset_index(drop=True)
    #     opt_price = pro.opt_daily(trade_date=trade_date,exchange="CFFEX")
    #     opt_price_temp = opt_price[opt_price['exchange'] == 'CFFEX'].reset_index(drop=True)
        opt_basic_temp = opt_basic[(opt_basic['ts_code'].isin(opt_price_temp['ts_code']))&(opt_basic['opt_code']==opt_code)][
            ['ts_code','call_put','maturity_date','exercise_price']
        ].reset_index(drop=True)
        
        opt_price = pd.merge(opt_price_temp,opt_basic_temp,how='inner')
        opt_price['tau'] = (pd.to_datetime(opt_price['maturity_date']) - pd.to_datetime(opt_price['trade_date'])).apply(lambda x:x.days)
        opt_price = opt_price.sort_values(by=['tau','exercise_price']).reset_index(drop=True)
        opt_price = opt_price[opt_price['tau']>7].reset_index(drop=True)
        opt_near = opt_price[opt_price['tau'] == opt_price['tau'].drop_duplicates().nsmallest(2).tolist()[0]].reset_index(drop=True)
        opt_2near = opt_price[opt_price['tau'] == opt_price['tau'].drop_duplicates().nsmallest(2).tolist()[1]].reset_index(drop=True)
        return opt_near,opt_2near
    
    
    # 根据CBOE公式计算SKEW
    @staticmethod
    def cal_S(opt_df,interp):
        opt_df_call = opt_df[opt_df['call_put'] == "C"].sort_values(by=['exercise_price']).reset_index(drop=True)
        opt_df_put = opt_df[opt_df['call_put'] == "P"].sort_values(by=['exercise_price']).reset_index(drop=True)
        
        opt_df_call = opt_df_call[['trade_date','call_put','tau','exercise_price','close']].drop_duplicates(['tau','exercise_price'],keep='first').rename(columns={'close':'call_close'})
        opt_df_put = opt_df_put[['trade_date','call_put','tau','exercise_price','close']].drop_duplicates(['tau','exercise_price'],keep='first').rename(columns={'close':'put_close'})
        
        opt_df_merge = pd.merge(opt_df_call,opt_df_put,on=['tau','exercise_price'])
        opt_df_merge['diff_close'] = abs(opt_df_merge['call_close'] - opt_df_merge['put_close'])
        S = opt_df_merge[opt_df_merge['diff_close'] == opt_df_merge['diff_close'].min()]['exercise_price'].iloc[0]
        
        tau = opt_df_merge['tau'].unique().tolist()[0]
        opt_df_merge['R'] = interp(tau)/100
        opt_df_merge['F'] = S + np.exp( opt_df_merge['R'] * (tau/365) )*(opt_df_merge[opt_df_merge['exercise_price'] == S]['call_close'] - opt_df_merge[opt_df_merge['exercise_price'] == S]['put_close']).iloc[0]#.mean()
        opt_df_merge['F - Ki'] = opt_df_merge['F'] - opt_df_merge['exercise_price']
        
        try:
            opt_df_merge['K0'] = opt_df_merge[ opt_df_merge['F - Ki'] ==  opt_df_merge['F - Ki'][opt_df_merge['F - Ki']>0].nsmallest(1).iloc[0]]['exercise_price'].iloc[0]
        except Exception:
            opt_df_merge['K0'] = opt_df_merge[ opt_df_merge['F - Ki'] ==  (opt_df_merge['F - Ki']).nsmallest(1).iloc[0]]['exercise_price'].iloc[0]
    
        opt_df_merge['PK'] = opt_df_merge.apply(lambda row: row['call_close'] if row['exercise_price']>row['K0'] else (row['put_close'] if row['exercise_price']<row['K0'] else (row['call_close']+row['put_close'])/2 ),axis=1)
        opt_df_merge['K_i+1'] = opt_df_merge['exercise_price'].shift(-1).fillna(opt_df_merge['exercise_price'].iloc[-1])
        opt_df_merge['K_i-1'] = opt_df_merge['exercise_price'].shift(1).fillna(opt_df_merge['exercise_price'].iloc[0])
        opt_df_merge['delta_K'] = (opt_df_merge['K_i+1'] - opt_df_merge['K_i-1'])/2
        opt_df_merge.loc[0,'delta_K'] = opt_df_merge.loc[0,'delta_K']*2
        opt_df_merge.loc[len(opt_df_merge)-1,'delta_K'] = opt_df_merge.loc[len(opt_df_merge)-1,'delta_K']*2
        #opt_df_merge = opt_df_merge.dropna().reset_index(drop=True)
    
        F = opt_df_merge['F'].iloc[0]
        K0 = opt_df_merge['K0'].iloc[0]
    
        opt_df_merge['p1'] = -np.exp(opt_df_merge['R']*opt_df_merge['tau']/365) * opt_df_merge['PK'] * opt_df_merge['delta_K']/(opt_df_merge['exercise_price']**2)
        P1 = opt_df_merge['p1'].sum() + (-1)*( 1 + np.log(F/K0) - F/K0 )
    
        opt_df_merge['p2'] = np.exp(opt_df_merge['R']*opt_df_merge['tau']/365) * 2 * (1-np.log(opt_df_merge['exercise_price']/opt_df_merge['F']))*opt_df_merge['PK'] * opt_df_merge['delta_K']/(opt_df_merge['exercise_price']**2)
        P2 = opt_df_merge['p2'].sum() + ( 2 * np.log(K0/F) * (F/K0 - 1) + 0.5 * np.log(K0/F)**2 )
    
        opt_df_merge['p3'] = np.exp(opt_df_merge['R']*opt_df_merge['tau']/365) * 3 * ( 2*np.log(opt_df_merge['exercise_price']/opt_df_merge['F']) - np.power(np.log(opt_df_merge['exercise_price']/opt_df_merge['F']),2)) * opt_df_merge['PK'] * opt_df_merge['delta_K'] / (opt_df_merge['exercise_price']**2)
        P3 = opt_df_merge['p3'].sum() + ( 3 * (np.log(K0/F)**2) * (1/3 * np.log(K0/F) - 1 + F/K0) )
    
        S = (P3 - 3*P1*P2 + 2*P1**3)/np.power((P2 - P1**2),3/2)
        return S
    
    # 对近月SKEW和次近月SKEW插值并年化得到ISKEW
    @staticmethod
    def cal_iskew(opt_df1,opt_df2,interp):
        tau1 = opt_df1['tau'].unique().tolist()[0]/365
        tau2 = opt_df2['tau'].unique().tolist()[0]/365
        if tau1 >=30:
            iskew = 100 - 10*OPTION.cal_S(opt_df1,interp)
        else:
            iskew = 100 - 10*(OPTION.cal_S(opt_df1,interp)*((tau2 - 30/365)/(tau2 - tau1)) + \
                    OPTION.cal_S(opt_df2,interp)*((30/365 - tau1)/(tau2 - tau1))) 
        #iskew = 100 - 10*cal_S(opt_df1)
        return iskew

if __name__ == '__main__':
    # data=xtdata.get_option_undl_data('510500.SH')
    pass

import os,sys,time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import traceback
import copy
import pickle
import multiprocessing as mp
import threading as td
import concurrent.futures
import random
import itertools
import math
from datetime import datetime, timedelta
import xlsxwriter
import argparse

from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from xtquant import xtdata
xtdata.enable_hello = False

np.random.seed(0)
# 添加项目路径=============================================================================
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,project_dir) 


def main(args):
    ####compare_rq_cty_size_factor 对比自己复现和米筐的size因子排序是否一致
    if args.task=='compare_rq_cty_size_factor':  
        print('compare_rq_cty_size_factor...')
        
        import rqdatac
        rqdatac.init()
        weights=rqdatac.index_weights(order_book_id='866006.RI', date=args.et)
        weights=weights.reset_index()
        weights['id']=list(weights['order_book_id'].apply(lambda id:rqdatac.id_convert(id,to='normal')))
        
        weights['close']=weights['id'].apply(lambda id: xtdata.get_instrument_detail(id)['PreClose'])
        # weights['floatvolume']=weights['id'].apply(lambda id: xtdata.get_instrument_detail(id)['FloatVolume'])
        weights['floatvolume']=weights['id'].apply(lambda id: xtdata.get_instrument_detail(id)['TotalVolume'])
        weights['size']=round(weights['close']*weights['floatvolume']/1e8,3)
        
        factor=rqdatac.get_factor_exposure(list(weights['order_book_id']), 
                                           args.et, args.et, factors = ['size'],
                                           industry_mapping='citics_2019', model = 'v2')
        
        size_sort=factor.sort_values('size',ascending=True)
        size_sort=size_sort.reset_index()
        # def callback(data):
        #     print(data)
        # xtdata.download_history_data2(list(weights['id']), period='1d', start_time='', end_time='', callback=callback,incrementally=True)    
 
    ####make_backtest_file3 自制因子
    if args.task=='make_backtest_file3':     
        import rqdatac
        rqdatac.init()
        inputs=[]
        # xtdata.download_holiday_data()
        calendar = xtdata.get_trading_calendar('SH', '20170101' , '20261231' )
        dates=calendar[calendar.index(args.st):calendar.index(args.et)+1]
        
        len_chosen=[]
        import statsmodels.formula.api as smf
        from scipy.stats.mstats import winsorize
        from sklearn.preprocessing import StandardScaler
            
        for date in tqdm(dates[::args.f]): 
            weights=[]
            for id in args.ids:
                part=xtdata.get_stock_list_in_sector('国证2000',date)
                weights+=part
            weights=set(weights)
            chosen=weights
            weights=pd.DataFrame(weights,columns=['id'])
            ###########################beta复现
            date_1=calendar[calendar.index(date)-1]
            date_252=calendar[calendar.index(date)-252]
            
            df1 = xtdata.get_market_data_ex(['close','preClose'],list(chosen),period='1d',
                                             start_time = date_252, end_time = date_1,
                                             count=-1,dividend_type='front_ratio')
            res1=[]
            for k,v in df1.items():
                if not v.isna().all().all():
                    v['id']=k
                    v.index.name='date'
                    v = v.reset_index().set_index(['id', 'date'])
                    res1.append(v)
            wpg=pd.concat(res1)
            
            df2 = xtdata.get_market_data_ex(['close','preClose'],['000300.SH'],period='1d',
                                             start_time = date_252, end_time = date_1,
                                             count=-1,dividend_type='front_ratio')
            res2=[]
            for k,v in df2.items():
                v['id']=k
                v.index.name='date'
                v = v.reset_index().set_index(['id', 'date'])
                res2.append(v)
            hs300=pd.concat(res2)
            
            hs300['return']=hs300['close']/hs300['preClose']-1
            hs300=hs300.bfill()
            hs300.index = hs300.index.get_level_values(1)
            def func(df,hs300):
                # print(df.index.get_level_values(0)[0])
                df.index = df.index.get_level_values(1)
                df['return']=df['close']/df['preClose']-1
                df=df.bfill()
                res=df[['return']].join(hs300[['return']], lsuffix='_wpg', rsuffix='_hs300')
                
                # 公式接口：y ~ x（因变量~自变量，默认添加截距项）
                model_ols = smf.ols(formula='return_wpg ~ return_hs300', data=res)
                result_ols = model_ols.fit()
                beta=result_ols.params.iloc[-1]
                
                return beta
            
            df=wpg.groupby(level=0).apply(func,hs300)
            df=pd.DataFrame(df,columns=['beta'])
            df['beta_winsorized'] = winsorize(df['beta'],limits=(0.03, 0.03))  # 上下截断比例：下5%和上5%
            df['beta_winsorized_01']=(df['beta_winsorized']-df['beta_winsorized'].min()) / (df['beta_winsorized'].max()-df['beta_winsorized'].min())
            
            ###########################size复现
            def func1(id):
                try:
                    return pd.Series([xtdata.get_instrument_detail(id)['PreClose'],xtdata.get_instrument_detail(id)['TotalVolume']])
                except:
                    return pd.Series([np.nan,np.nan])
            df2=weights.copy()
            df2[['close','TotalVolume']]=df2['id'].apply(func1)
            df2=df2.dropna()
            df2['market_value']=round(df2['close']*df2['TotalVolume']/1e8,3)
            df2['size_winsorized'] = winsorize(df2['market_value'],limits=(0.03, 0.03))  # 上下截断比例：下5%和上5%
            df2['size_winsorized_01']=(df2['size_winsorized']-df2['size_winsorized'].min()) / (df2['size_winsorized'].max()-df2['size_winsorized'].min())
            
            ############################liquidity复现
            date_1=calendar[calendar.index(date)-1]
            date_252=calendar[calendar.index(date)-252]
            
            df3 = xtdata.get_market_data_ex(['amount'],list(chosen),period='1d',
                                             start_time = date_252, end_time = date_1,
                                             count=-1,dividend_type='front_ratio')
            market_dic = df2.set_index("id")["market_value"].to_dict()
            res3=[]
            for k,v in df3.items():
                # if not v.isna().all().all():
                try:
                    v['id']=k
                    v['amount']=round(v['amount']/1e8,3)
                    v['turnover']=v['amount']/market_dic[v['id'].iloc[0]]
                    week_turnover=v['turnover'].iloc[-7:].mean()
                    month_turnover=v['turnover'].iloc[-30:].mean()
                    year_turnover=v['turnover'].mean()
                    res3.append([k,week_turnover,month_turnover,year_turnover])
                except:
                    print(f'wrong:{k}')
            df3=pd.DataFrame(res3,columns=['id','week_turnover','month_turnover','year_turnover'])
            df3['liquidity']=0.35*df3['week_turnover']+0.35*df3['month_turnover']+0.3*df3['year_turnover']
            df3['liquidity_winsorized'] = winsorize(df3['liquidity'],limits=(0.03, 0.03))  # 上下截断比例：下5%和上5%
            df3['liquidity_winsorized_01']=(df3['liquidity_winsorized']-df3['liquidity_winsorized'].min()) / (df3['liquidity_winsorized'].max()-df3['liquidity_winsorized'].min())
            
            res=df[['beta_winsorized_01']].join(df2[['size_winsorized_01']]).join(df3[['liquidity_winsorized_01']])
            res.columns=['beta','size','liquidity']
            # 计算综合评分（值越高越符合目标）
            res['score']=0
            for factor in list(args.factors.keys()):
                res['score']+=res[factor] *args.factors[factor]
            res=res.sort_values('score',ascending=False)
            chosen=set(res.iloc[:args.top_n].index)
            
            
            weights=weights[weights['id'].isin(chosen)]
            weights.columns=['TICKER']
            weights['TRADE_DT']=date
            weights['NAME']=weights['TICKER'].apply(lambda id:xtdata.get_instrument_detail(id)['InstrumentName'])
            weights=weights[['TRADE_DT','TICKER','NAME']]
            weights['TARGET_WEIGHT']=1/len(weights)
            weights=weights.reset_index()
            inputs.append(weights)
        inputs=pd.concat(inputs,axis=0)
        with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
            inputs.to_excel(writer, sheet_name='', index=False)  
            print(f'save to {args.file}')        
        
        x=a

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    ####入参
    
    ####常用指数
    # args.task='compare_rq_cty_size_factor'
    # args.et=20250915
    
    # args.task='compare_rq_cty_liquidity_factor'
    # args.et=20250915
    
    args.task='make_backtest_file3' ##自制因子
    args.ids=[
              # '000852.XSHG',
              # '932000.INDX',
              # '399303.XSHE',
              '国证2000'
              # '866006.RI',
              ]
    
    args.factors={
        'size':-0.5,
        'beta':0.5,
        'liquidity':-0.5,
        }
    args.top_n=100
    args.st='20240923'
    args.et='20251016'
    args.days=1   ##因子回看天数
    args.f=5   ##调仓频率
    args.file=r'data/增强等权周频.xlsx'
    
    main(args)
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
import uuid

import rqdatac
from rqalpha_plus import *
from rqalpha.apis import *
import rqalpha
import rqalpha_mod_fund
from rqoptimizer import *
import talib
rqdatac.init()
from alphalens.utils import get_clean_factor_and_forward_returns
import alphalens

from config.config import CS,INDX
from tools.convert_func import Convert  
from tools.metrics_func import Metrics
from tools.general_func import General
# from tools.factor_func import Factor
# from tools.option_func import OPTION
from tools.plot_func import Plot
# from tools.analysis_func import Analysis
from tools.riskfolio_func import Riskfolio

np.random.seed(0)

def main(args):

    ####!!!!!!!generate_target_pos 根据任意池子+任意因子等权生成dfcf目标持仓
    if args.task=='generate_target_pos':  
        print('generate_target_pos...')
        
        weights=[]
        for id in args.ids:
            part=rqdatac.index_weights(order_book_id=id, date=args.et)
            weights.append(part)
        weights=pd.concat(weights)
        weights=weights.reset_index()
        weights = weights.drop_duplicates(subset=['order_book_id'], keep='last')
        
        #过滤被立案的
        announcement=rqdatac.get_announcement(list(weights['order_book_id']),'20240101',args.et)
        cc=announcement[announcement['title'].str.contains('立案')]
        cc=cc.reset_index()
        cc=set(cc['order_book_id'])
        weights=weights[~weights['order_book_id'].isin(cc)]        
        
        exposure=rqdatac.get_factor_exposure(list(weights['order_book_id']), 
                                           args.et, args.et, factors = list(args.factors.keys()),
                                           industry_mapping='citics_2019', model = 'v2')
        # 计算综合评分（值越高越符合目标）
        exposure['score']=0
        for factor in list(args.factors.keys()):
            exposure['score']+=exposure[factor] *args.factors[factor]
            
        exposure=exposure.sort_values('score',ascending=False)
        chosen=set(exposure.iloc[:args.top_n].index.get_level_values(1))
        df=weights[weights['order_book_id'].isin(chosen)]
        number=[]
        for id in args.ids:   ##成分股占比监测
            part=rqdatac.index_weights(order_book_id=id, date=args.et)
            n=len(df[df['order_book_id'].isin(list(part.index))])
            number.append([id,n])
        print(number)
        
        
        df.columns=['id','weight']
        df['weight']=1/len(df) ##重新计算权重
        df['买卖日期']=args.et.strftime('%Y-%m-%d')
        df['证券代码']=[i for i in rqdatac.id_convert(list(df['id']),to='normal')]
        df['name']=[i.symbol for i in rqdatac.instruments(list(df['id']), market='cn')]
        df['买卖价格']=list(rqdatac.get_price(order_book_ids=list(df['id']), 
                  start_date=args.et, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)['close'])   
        each=args.money/len(df)
        df['买卖数量']=df['买卖价格'].apply(lambda close:int(each//(close*100)*100))
        df.loc[(df['证券代码'].str.startswith('688')) & (df['买卖数量'] == 100), '买卖数量'] = 200 ##科创板至少200股
        df['买卖方向']='买入'
        res=df[['买卖日期','证券代码', '买卖数量', '买卖价格', '买卖方向']]
        res=res[res['买卖数量']!=0] ##去掉价格过高，分配不到100股的票
        money=sum(res['买卖价格']*res['买卖数量'])
        
        buy=res[res['买卖方向']=='买入']
        if len(buy)!=0:
            buy_money=sum(buy['买卖价格']*buy['买卖数量'])
        else:
            buy_money=0
            
        sell=res[res['买卖方向']=='卖出']
        if len(sell)!=0:
            sell_money=sum(sell['买卖价格']*sell['买卖数量'])
        else:
            sell_money=0        
        print('总数:',money)
        print('买入:',buy_money)
        print('买入笔数:',len(buy))
        print('卖出:',sell_money)
        print('卖出笔数:',len(sell))
        
        
        now = args.et.strftime("%Y-%m-%d")##文件名和时间有关联
        path=f'./DFCF_csv/篮子/目标篮子_{now}.csv'  
        
        res=res[['证券代码', '买卖方向', '买卖数量','买卖价格']]  
        res.columns=['代码', '交易方向', '数量','委托价格']      
        res['代码']=res['代码'].apply(lambda x:x.split('.')[0])
        res['交易方向']=res['交易方向'].apply(lambda x:1 if x=='买入' else 2)
        res['权重']=np.nan
        # res['委托价格']=np.nan
        res['基准价格']=np.nan
        
        res.to_csv(path,index=False)
        print(f'save to {path}')
        
    ####!!!!!!!adjust_pos 根据持仓篮子+目标篮子，生成调仓篮子
    if args.task=='adjust_pos':  
        print('adjust_pos...')
        
        hold_pos=pd.read_csv(args.hold_pos,encoding='gbk',dtype=str) ##现有持仓
        target_pos=pd.read_csv(args.target_pos,dtype=str) ##目标持仓
        
        hold_pos['代码']=hold_pos['symbol'].apply(lambda x:x.split('.')[-1])
        df=pd.merge(hold_pos[['代码','availableNow','lastPrice','marketValue','fpnl']],
                    target_pos[['代码','数量','委托价格']],on='代码',how='outer')
        df['lastPrice'].fillna(df['委托价格'], inplace=True)
        del df['委托价格']
        df.columns=['代码', '现有持仓', '价格', '市值', '盈亏', '目标持仓']
        df=df.fillna(0)
        df['调整股数']=df['目标持仓'].apply(int)-df['现有持仓'].apply(int)
        df=df.sort_values('调整股数',ascending=True)
        df['调整金额']=df['调整股数'].apply(float)*df['价格'].apply(float)
        def func1(row):##保留所需行
            if abs(row['调整金额'])>args.adjust_threshold: #调整幅度太小的，没有免5不划算，其它保留
                return True
            else:
                return False
        df['是否保留']=df.apply(func1,axis=1)
            
        res=df[df['是否保留']==True]
        
        def func2(row): ##处理科创板
            if row['代码'].startswith('688') and row['调整股数']>0 and row['调整股数']<200: ##科创板
                return 200
            else:
                return row['调整股数']
        res['调整股数']=res.apply(func2,axis=1)
        
        money=res['调整金额'].apply(abs).sum()
        buy=res[res['调整金额']>0]
        if len(buy)!=0:
            buy_money=buy['调整金额'].sum()
        else:
            buy_money=0
            
        sell=res[res['调整金额']<0]
        if len(sell)!=0:
            sell_money=sell['调整金额'].sum()
        else:
            sell_money=0        
        print('总数:',money)
        print('买入:',buy_money)
        print('买入笔数:',len(buy))
        print('卖出:',sell_money)
        print('卖出笔数:',len(sell))
        
        
        res['交易方向']=res['调整股数'].apply(lambda x:1 if x>0 else 2)
        res['数量']=res['调整股数'].apply(abs)
        res=res[['代码', '交易方向', '数量']]  
        res['权重']=np.nan
        res['委托价格']=np.nan
        res['基准价格']=np.nan
        
        res=res.reset_index(drop=True)
        path=f'./DFCF_csv/篮子/调仓篮子_{args.et}.csv'  
        res.to_csv(path,index=False)
        print(f'save to {path}')

    ####!!!!!!!trade_log_to_pms_real 实盘！！！！根据成交记录，变成pms追踪
    if args.task=='trade_log_to_pms_real':  
        print('trade_log_to_pms_real...')
        
        df=pd.read_csv(args.trade_log)
        df=df[['created_at','symbol','price','volume','side']]
        df.columns=['买卖日期','证券代码', '买卖价格', '买卖数量', '买卖方向']
        df['买卖日期']=args.et
        df['证券代码']=df['证券代码'].apply(lambda x:x.split('.')[1]+'.'+x.split('.')[0][:2])
        df['买卖方向']=df['买卖方向'].apply(lambda x:'买入' if x==1 else '卖出')
        
        cash = {'证券代码': 'CNY', '买卖数量': np.nan, '买卖价格': np.nan, '买卖方向': '划入'}            
        future = {'证券代码': 'IM2510.CFE', '买卖数量': np.nan, '买卖价格': np.nan, '买卖方向': '空开'}            
        cash=pd.DataFrame([cash])
        future=pd.DataFrame([future])
        res=pd.concat([cash,future,df])   
        
        acc='DFCF'  ##文件名和账户名有关联
        now = res['买卖日期'].iloc[-1]
        path=f'./PMS_csv/{acc}_{now}.xlsx'  
        with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
            res.to_excel(writer, sheet_name='导入数据区', index=False)   
            print(f'save to {path}')        
        
        xx=1
    ####!!!!!!!adjust_pos 根据持仓篮子+目标篮子，生成调仓篮子
    if args.task=='adjust_pos':  
        print('adjust_pos...')
        
        hold_pos=pd.read_csv(args.hold_pos,encoding='gbk',dtype=str) ##现有持仓
        target_pos=pd.read_csv(args.target_pos,dtype=str) ##目标持仓
        
        hold_pos['代码']=hold_pos['symbol'].apply(lambda x:x.split('.')[-1])
        df=pd.merge(hold_pos[['代码','availableNow','lastPrice','marketValue','fpnl']],
                    target_pos[['代码','数量','委托价格']],on='代码',how='outer')
        df['lastPrice'].fillna(df['委托价格'], inplace=True)
        del df['委托价格']
        df.columns=['代码', '现有持仓', '价格', '市值', '盈亏', '目标持仓']
        df=df.fillna(0)
        df['调整股数']=df['目标持仓'].apply(int)-df['现有持仓'].apply(int)
        df=df.sort_values('调整股数',ascending=True)
        df['调整金额']=df['调整股数'].apply(float)*df['价格'].apply(float)
        def func1(row):##保留所需行
            if abs(row['调整金额'])>args.adjust_threshold: #调整幅度太小的，没有免5不划算，其它保留
                return True
            else:
                return False
        df['是否保留']=df.apply(func1,axis=1)
            
        res=df[df['是否保留']==True]
        
        def func2(row): ##处理科创板
            if row['代码'].startswith('688') and row['调整股数']>0 and row['调整股数']<200: ##科创板
                return 200
            else:
                return row['调整股数']
        res['调整股数']=res.apply(func2,axis=1)
        
        money=res['调整金额'].apply(abs).sum()
        buy=res[res['调整金额']>0]
        if len(buy)!=0:
            buy_money=buy['调整金额'].sum()
        else:
            buy_money=0
            
        sell=res[res['调整金额']<0]
        if len(sell)!=0:
            sell_money=sell['调整金额'].sum()
        else:
            sell_money=0        
        print('总数:',money)
        print('买入:',buy_money)
        print('买入笔数:',len(buy))
        print('卖出:',sell_money)
        print('卖出笔数:',len(sell))
        
        
        res['交易方向']=res['调整股数'].apply(lambda x:1 if x>0 else 2)
        res['数量']=res['调整股数'].apply(abs)
        res=res[['代码', '交易方向', '数量']]  
        res['权重']=np.nan
        res['委托价格']=np.nan
        res['基准价格']=np.nan
        
        res=res.reset_index(drop=True)
        path=f'./DFCF_csv/篮子/调仓篮子_{args.et}.csv'  
        res.to_csv(path,index=False)
        print(f'save to {path}')

    ####!!!!!!!trade_log_to_pms 根据成交记录，变成pms追踪
    if args.task=='trade_log_to_pms':  
        print('trade_log_to_pms...')
        
        df=pd.read_csv(args.trade_log)
        df=df[['created_at','symbol','price','volume','side']]
        df.columns=['买卖日期','证券代码', '买卖价格', '买卖数量', '买卖方向']
        df['买卖日期']=df['买卖日期'].apply(lambda x:x.split('T')[0])
        df['证券代码']=df['证券代码'].apply(lambda x:x.split('.')[1]+'.'+x.split('.')[0][:2])
        df['买卖方向']=df['买卖方向'].apply(lambda x:'买入' if x==1 else '卖出')
        
        cash = {'证券代码': 'CNY', '买卖数量': np.nan, '买卖价格': np.nan, '买卖方向': '划入'}            
        future = {'证券代码': 'IM2510.CFE', '买卖数量': np.nan, '买卖价格': np.nan, '买卖方向': '空开'}            
        cash=pd.DataFrame([cash])
        future=pd.DataFrame([future])
        res=pd.concat([cash,future,df])   
        
        acc='DFCF'  ##文件名和账户名有关联
        now = res['买卖日期'].iloc[-1]
        path=f'./PMS_csv/{acc}_{now}.xlsx'  
        with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
            res.to_excel(writer, sheet_name='导入数据区', index=False)   
            print(f'save to {path}')        
        
        xx=1
        
    ####!!!!!!!basis_detect 基差监测
    if args.task=='basis_detect':  
        print('basis_detect...')
        basis=rqdatac.futures.get_basis(order_book_ids=args.id, 
                                        start_date=args.st,end_date=args.et,
                                        fields=None,frequency='1d')
        print('days:',len(basis))
        basis['basis'].plot()

    ####!!!!!!!!cty_select 题材选股
    if args.task=='cty_select':   
        df=pd.read_csv('data/人形机器人优选股.csv')
        df=df.iloc[:-1]
        df['证券代码']=[i for i in rqdatac.id_convert(list(df['id']),to=None)]
        df['weight']=1/len(df) ##重新计算权重
        df['买卖日期']=args.et.strftime('%Y-%m-%d')
        df['买卖价格']=list(rqdatac.get_price(order_book_ids=list(df['证券代码']), 
                  start_date=args.et, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)['close'])   
        
        each=args.money/len(df)
        df['买卖数量']=df['买卖价格'].apply(lambda close:int(each//(close*100)*100))
        df.loc[(df['证券代码'].str.startswith('688')) & (df['买卖数量'] == 100), '买卖数量'] = 200 ##科创板至少200股
        df['买卖方向']='买入'
        res=df[['买卖日期','证券代码', '买卖数量', '买卖价格', '买卖方向']]
        res=res[res['买卖数量']!=0] ##去掉价格过高，分配不到100股的票
        money=sum(res['买卖价格']*res['买卖数量'])
        
        buy=res[res['买卖方向']=='买入']
        if len(buy)!=0:
            buy_money=sum(buy['买卖价格']*buy['买卖数量'])
        else:
            buy_money=0
            
        sell=res[res['买卖方向']=='卖出']
        if len(sell)!=0:
            sell_money=sum(sell['买卖价格']*sell['买卖数量'])
        else:
            sell_money=0        
        print('总数:',money)
        print('买入:',buy_money)
        print('买入笔数:',len(buy))
        print('卖出:',sell_money)
        print('卖出笔数:',len(sell))
        
        now = args.et.strftime("%Y-%m-%d")##文件名和时间有关联
        path=f'./DFCF_csv/篮子/目标篮子_{now}.csv'  
        
        res=res[['证券代码', '买卖方向', '买卖数量','买卖价格']]  
        res.columns=['代码', '交易方向', '数量','委托价格']      
        res['代码']=res['代码'].apply(lambda x:x.split('.')[0])
        res['交易方向']=res['交易方向'].apply(lambda x:1 if x=='买入' else 2)
        res['权重']=np.nan
        # res['委托价格']=np.nan
        res['基准价格']=np.nan
        
        res.to_csv(path,index=False)
        print(f'save to {path}')
        
        
        xx=1
        
if __name__ == '__main__':
    ####入参
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.task='make_order'
    
    args.task='basis_detect'
    args.id='IM2409'
    args.st=20240101
    args.et=20240901
    
    # args.task='adjust_pos'
    # args.hold_pos='DFCF_csv/持仓/positions.csv'
    # args.target_pos='DFCF_csv/篮子/目标篮子_2025-10-28.csv'
    # args.adjust_threshold=1e4 ##调整金额小于阈值的就不调整了，因为不免5
    # args.et='2025-10-29'
    
    # args.task='generate_target_pos'     
    # args.ids=[
    #           # '000852.XSHG',
    #           # '932000.INDX',
    #           '399303.XSHE',
    #           # '866006.RI',
    #           ]
    
    # args.factors={
    #     'size':-0.5,
    #     'beta':0.5,
    #     'liquidity':-0.5,
    #     }
    # args.top_n=100
    # args.et=pd.to_datetime('20251028')
    # args.money=158e4

    # args.task='trade_log_to_pms'     
    # args.trade_log='DFCF_csv/成交/主观成交25-11-07.csv'
    
    # args.task='trade_log_to_pms_real'       ##实盘版！！
    # args.trade_log='DFCF_csv/成交/主观成交25-11-07.csv'
    # args.et='2025-11-07'
    
    # args.task='cty_select'
    # args.et=pd.to_datetime('20251106')
    # args.money=55e4
    
    main(args)
    
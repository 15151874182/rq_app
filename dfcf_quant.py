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
    ####make_order ##制作.csv文件
    if args.task=='make_order': 
        print('make_order')
        
        cols=['sid', 'account_id', 'symbol', 'volume', 'order_type', 'order_business(order_biz)', 'price', 'comment']
        res=[]
        res.append([f'{uuid.uuid4().hex}','e2cdd75b-8e51-11f0-b2eb-52560acd7da0','SZSE.000001','1000','24','2','','测试'])
        res.append([f'{uuid.uuid4().hex}','e2cdd75b-8e51-11f0-b2eb-52560acd7da0','SHSE.600000','1000','24','2','','测试'])
        res=pd.DataFrame(res,columns=cols)
        res.to_csv('DFCF_csv/input/test.order2.csv',index=False)
        xx=1

    ####rq_wpg_make_pms_csv2 根据米筐微盘成分股等权生成pms目标持仓清单,liq
    if args.task=='rq_wpg_make_pms_csv2':  
        print('rq_wpg_make_pms_csv2...')
        df=rqdatac.index_weights(order_book_id=args.id, date=args.et) ##中证500
        df=df.reset_index()
        df.columns=['id','weight']
        
        # ##过滤被立案的
        # announcement=rqdatac.get_announcement(list(df['id']),'20240101',args.et)
        # cc=announcement[announcement['title'].str.contains('立案')]
        # cc=cc.reset_index()
        # cc=set(cc['order_book_id'])
        # df=df[~df['id'].isin(cc)] 
        
        factor=rqdatac.get_factor_exposure(list(df['id']), 
                                           args.et, args.et, factors = ['liquidity'],
                                           industry_mapping='citics_2019', model = 'v2')
        
        liquidity_sort=factor.sort_values('liquidity',ascending=True)
        liquidity_sort=liquidity_sort.reset_index()
        
        chosen=set(liquidity_sort['order_book_id'].iloc[:args.n])
            
            
        df=df[df['id'].isin(chosen)]
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
        
        acc='仿真篮子'  ##文件名和账户名有关联
        now = args.et.strftime("%Y-%m-%d")##文件名和时间有关联
        path=f'./DFCF_csv/篮子/{acc}_{now}.csv'  
        
        res=res[['证券代码', '买卖方向', '买卖数量']]  
        res.columns=['代码', '交易方向', '数量']      
        res['代码']=res['代码'].apply(lambda x:x.split('.')[0])
        res['交易方向']=res['交易方向'].apply(lambda x:1 if x=='买入' else 2)
        res['权重']=np.nan
        res['委托价格']=np.nan
        res['基准价格']=np.nan
        
        res.to_csv(path,index=False)
        print(f'save to {path}')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.task='make_order'
    
    
    args.task='rq_wpg_make_pms_csv2'     ##liq
    # args.id='000002.XSHG' ##全A
    # args.id='000300.XSHG' ##沪深300
    # args.id='399852.XSHE' ##中证1000
    # args.id='000905.XSHG' ##中证500
    args.id='866006.RI'
    args.et=pd.to_datetime('20250912')
    args.money=200e4
    args.n=300 ##top多少票
    
    main(args)
    
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

    ####!!!!!!!generate_target_pos 百榕定制版，1000+2000+微盘分别，生成ATX目标持仓
    if args.task=='generate_target_pos':  
        print('generate_target_pos...')
        
        df=[]
        dic={'000852.XSHG':80,'932000.INDX':120,'866006.RI':120}
        for id in dic.keys():
            weights=rqdatac.index_weights(order_book_id=id, date=args.et)
            weights=weights.reset_index()
            
            #过滤被立案的
            announcement=rqdatac.get_announcement(list(weights['order_book_id']),'20250101',args.et)
            cc=announcement[announcement['title'].str.contains('立案')]
            cc=cc.reset_index()
            cc=set(cc['order_book_id'])
            weights=weights[~weights['order_book_id'].isin(cc)]        
            
            ###剔除夕阳行业版本
            exposure=rqdatac.get_factor_exposure(list(weights['order_book_id']), 
                                               args.et, args.et, factors = None,
                                               # date_1, date_1, factors = list(args.factors.keys()),
                                               industry_mapping='citics_2019', model = 'v2')
            for col in ['建材','建筑','房地产','纺织服装','综合']: ##去掉这些夕阳行业
                exposure = exposure[exposure[col] != 1]
            # 计算综合评分（值越高越符合目标）
            exposure['score']=0
            for factor in list(args.factors.keys()):
                exposure['score']+=exposure[factor] *args.factors[factor]
                
            exposure=exposure.sort_values('score',ascending=False)
            chosen=set(exposure.iloc[:dic[id]].index.get_level_values(1))
            weights=weights[weights['order_book_id'].isin(chosen)]
            df.append(weights)
        df=pd.concat(df)
        df = df.drop_duplicates(subset=['order_book_id'], keep='last')
        
        df['weight']=1/len(df) ##重新计算权重
        df['买卖日期']=args.et2.strftime('%Y-%m-%d')
        df['证券代码']=[i for i in rqdatac.id_convert(list(df['order_book_id']),to='normal')]
        df['name']=[i.symbol for i in rqdatac.instruments(list(df['order_book_id']), market='cn')]
        df['买卖价格']=list(rqdatac.get_price(order_book_ids=list(df['order_book_id']), 
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
        
        res['算法类型']='TWAP'
        res['账户名称']=args.account
        res['算法实例']='kf_twap_plus'
        res['证券代码']=res['证券代码']
        res['交易方向']=res['买卖方向']
        res['任务数量']=res['买卖数量'].apply(abs)
        res['开始时间']=args.start_time
        res['结束时间']=args.end_time
        res['涨跌停是否继续执行']='涨停不卖跌停不买'
        res['过期后是否继续执行']='否'
        res['其他参数']=np.nan
        res['交易市场']=np.nan
        
        res=res.reset_index()
        
        columns=['算法类型',
                '账户名称',
                '算法实例',
                '证券代码',
                '任务数量',
                '交易方向',
                '开始时间',
                '结束时间',
                '涨跌停是否继续执行',
                '过期后是否继续执行',
                '其他参数',
                '交易市场']
        res=res[columns]
        
        res.to_csv(args.ATX_file,index=False)
        print(f'save to {args.ATX_file}')                

    ####！！！！rq_wpg_adjust_ATX 根据ATX的实时监控.xlsx和米筐的微盘股目标仓位，生成csv，用于ATX 调仓
    if args.task=='rq_wpg_adjust_ATX':  
        print('rq_wpg_adjust_ATX...')
        
        df1=pd.read_excel(args.ATX_pos_file,dtype={'证券代码': str}) ##现有持仓
        df2=pd.read_excel(args.pms_file, sheet_name='导入数据区') ##目标持仓
        
        df1['证券市场']=df1['交易市场'].apply(lambda x:'SZ' if x=='深交所' else 'SH')
        df1['证券代码']=df1['证券代码']+'.'+df1['证券市场']
        df1['当前拥股']=df1['持仓数量']
        df1=df1[['证券代码','当前拥股']]
        
        # df2=df2.iloc[1:,:]
        df2=df2[['证券代码','买卖数量']]
        df2.columns=['证券代码','目标拥股']        
        df=pd.merge(df1,df2,on='证券代码',how='outer')
        df=df.fillna(0)
        df['调整股数']=(df['目标拥股']-df['当前拥股']).apply(int)
        df=df.sort_values('调整股数',ascending=True)
        df=df[df['调整股数']!=0]
        
        def func(row):
            if row['调整股数']>0 and row['调整股数']<200:
                if row['证券代码'].startswith('688'): ##科创板
                    return 200
                else:
                    return 100
            else:
                return row['调整股数']
                
        df['调整股数']=df.apply(func,axis=1)
        
        df['算法类型']='TWAP'
        df['账户名称']=args.account
        df['算法实例']='kf_twap_plus'
        df['证券代码']=df['证券代码']
        df['交易方向']=df['调整股数'].apply(lambda x:'买入' if x>0 else '卖出')
        df['任务数量']=df['调整股数'].apply(abs)
        df['开始时间']=args.start_time
        df['结束时间']=args.end_time
        # df['开始时间']='20241219T093000000'
        # df['结束时间']='20241219T103000000'
        df['涨跌停是否继续执行']='涨停不卖跌停不买'
        df['过期后是否继续执行']='否'
        df['其他参数']=np.nan
        df['交易市场']=np.nan
        
        df=df.reset_index()
        
        df.loc[(df['证券代码'].str.startswith('688')) & (df['调整股数'] == 100), '调整股数'] = 0 ##科创板至少200股,就不调整了
        df.loc[(df['证券代码'].str.startswith('688')) & (df['调整股数'] == -100), '调整股数'] = 0 ##科创板至少200股,就不调整了
        df=df[df['调整股数']!=0]
        
        columns=['算法类型',
                '账户名称',
                '算法实例',
                '证券代码',
                '任务数量',
                '交易方向',
                '开始时间',
                '结束时间',
                '涨跌停是否继续执行',
                '过期后是否继续执行',
                '其他参数',
                '交易市场']
        df=df[columns]
        
        df.to_csv(args.ATX_file,index=False)
        print(f'save to {args.ATX_file}')

    ####!!!!!!!generate_target_pos2 (米筐风险模型不能用的情况)根据任意池子+任意因子等权生成dfcf目标持仓
    if args.task=='generate_target_pos2':  
        print('generate_target_pos2...')
        import statsmodels.formula.api as smf
        from scipy.stats.mstats import winsorize
        from sklearn.preprocessing import StandardScaler
        
        weights=[]
        for id in args.ids:
            part=rqdatac.index_weights(order_book_id=id, date=args.et)
            weights.append(part)
        weights=pd.concat(weights)
        weights=weights.reset_index()
        weights = weights.drop_duplicates(subset=['order_book_id'], keep='last')
        
        ###########################beta复现
        date_252=rqdatac.get_previous_trading_date(args.et,n=252,market='cn')
        wpg=rqdatac.get_price(order_book_ids=list(weights['order_book_id']), 
                  start_date=date_252, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        hs300=rqdatac.get_price(
                  order_book_ids='000300.XSHG', 
                  start_date=date_252, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        
           
        hs300['return']=hs300['close']/hs300['close'].shift(1)-1
        hs300=hs300.bfill()
        hs300.index = hs300.index.get_level_values(1)
        def func(df,hs300):
            df.index = df.index.get_level_values(1)
            df['return']=df['close']/df['close'].shift(1)-1
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
        shares=rqdatac.get_shares(list(weights['order_book_id']), start_date=args.et, end_date=args.et, fields=None, market='cn', expect_df=True)['total_a'] 
        shares.index = shares.index.get_level_values(0)

        close=rqdatac.get_price(order_book_ids=list(weights['order_book_id']), 
                  start_date=args.et, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='none', skip_suspended =False, market='cn',  ##要不复权的价格！！！
                  expect_df=True,time_slice=None)['close']
        close.index = close.index.get_level_values(0)
        df2=weights.set_index('order_book_id')
        shares = shares.to_frame(name='total_share') 
        close = close.to_frame(name='close') 
        df2=df2.join(shares).join(close)
        
        df2['market_value']=df2['close']*df2['total_share'] ##总市值
        df2['size_winsorized'] = winsorize(df2['market_value'],limits=(0.03, 0.03))  # 上下截断比例：下5%和上5%
        df2['size_winsorized_01']=(df2['size_winsorized']-df2['size_winsorized'].min()) / (df2['size_winsorized'].max()-df2['size_winsorized'].min())
        
        ############################liquidity复现
        turnover=rqdatac.get_turnover_rate(list(weights['order_book_id']), 
                                           start_date=args.et, end_date=args.et, 
                                           fields=None,expect_df=True)
        turnover.index = turnover.index.get_level_values(0)
        turnover = turnover[~(turnover == 0).all(axis=1)]
        turnover['liquidity']=0.35*turnover['week']+0.35*turnover['month']+0.3*turnover['year']
        turnover['liquidity_winsorized'] = winsorize(turnover['liquidity'],limits=(0.03, 0.03))  # 上下截断比例：下5%和上5%
        turnover['liquidity_winsorized_01']=(turnover['liquidity_winsorized']-turnover['liquidity_winsorized'].min()) / (turnover['liquidity_winsorized'].max()-turnover['liquidity_winsorized'].min())
        
        res=df[['beta_winsorized_01']].join(df2[['size_winsorized_01']]).join(turnover[['liquidity_winsorized_01']])
        res.columns=['beta','size','liquidity']
        # 计算综合评分（值越高越符合目标）
        res['score']=0
        for factor in list(args.factors.keys()):
            res['score']+=res[factor] *args.factors[factor]
        res=res.sort_values('score',ascending=False)
        chosen=set(res.iloc[:args.top_n].index)
        
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
        df=df[['createdAt','symbol','price','volume','side']]
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

        
if __name__ == '__main__':
    ####入参
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.task='generate_target_pos'     
    args.factors={
        'size':-0.5,
        'beta':0.5,
        'liquidity':-0.5,
        }
    args.et=pd.to_datetime('20260126') ##交易日前收盘日，eg周一交易，这里写周五
    args.et2=pd.to_datetime('20260127') ##待交易日期，eg周一要买卖，写周一
    args.money=400e4
    args.start_time='20260127T093000000'
    args.end_time=  '20260127T093500000'  
    args.ATX_file='ATX_csv/ATX_stock_2026-01_27百里挑一信用.csv'
    args.account='百榕百里挑一稳健一号信用'
    
    # args.task='rq_wpg_adjust_ATX'
    # args.pms_file='PMS_csv/共同target_2026-01-20.xlsx' ##目标持仓
    # args.start_time='20260127T093000000'
    # args.end_time=  '20260127T093500000'  
    # args.ATX_pos_file='ATX_csv/持仓查询none.xlsx'  ##现有持仓
    # args.ATX_file='ATX_csv/ATX_stock_2026-01_27百里挑一信用.csv'
    # args.account='百榕百里挑一稳健一号信用'
    # # args.ATX_pos_file='ATX_csv/全天候持仓查询_20250901090843.xlsx'
    # # args.ATX_file='ATX_csv/ATX_stock_2025-09-01_绝对收益信用.csv'
    # # args.account='百榕全天候宏观对冲绝对收益信用'
    
    # args.task='generate_target_pos2'     ###不依赖付费风险模型
    # args.ids=[
    #           # '000852.XSHG',
    #           # '932000.INDX',
    #           # '399303.XSHE',
    #           '866006.RI',
    #           ]
    
    # args.factors={
    #     'size':-0.5,
    #     'beta':0.5,
    #     'liquidity':-0.5,
    #     }
    # args.top_n=150
    # args.et=pd.to_datetime('20251211')
    # args.money=190e4

    # args.task='trade_log_to_pms'     
    # args.trade_log='DFCF_csv/成交/execution_report.csv'
    
    # args.task='trade_log_to_pms_real'       ##实盘版！！
    # args.trade_log='DFCF_csv/成交/2量化成交26-01-21.csv'
    # args.et='2026-01-21'
    
    # args.task='cty_select'
    # args.et=pd.to_datetime('20251106')
    # args.money=55e4
    
    main(args)
    
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

import rqdatac
from rqalpha_plus import *
from rqalpha.apis import *
import rqalpha
import rqalpha_mod_fund
from rqoptimizer import *
import talib
rqdatac.init()

from config.config import CS,INDX
from tools.convert_func import Convert  
from tools.metrics_func import Metrics
from tools.general_func import General
# from tools.factor_func import Factor
# from tools.option_func import OPTION
from tools.plot_func import Plot
# from tools.analysis_func import Analysis
# from tools.riskfolio_func import Riskfolio

np.random.seed(0)
# 添加项目路径=============================================================================
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,project_dir) 


def main(args):
    ####make_config ##制作config/CS.csv文件
    if args.task=='make_config': 
        CS=rqdatac.all_instruments(type='CS', market='cn', date=None)
        INDX=rqdatac.all_instruments(type='INDX', market='cn', date=None)
        
        CS.to_csv('config/CS.csv',index=False) 
        INDX.to_csv('config/INDX.csv',index=False) 

    ####stratgy1 策略1
    if args.task=='stratgy1':      
        cols_risk_factor=[
            # 市场风险因子
            'beta', 
            'residual_volatility',
        
            # 价值/基本面因子
            'book_to_price', 
            'dividend_yield', 
            'earnings_yield',
        
            # 质量/盈利因子
            'earnings_quality', 
            'profitability', 
            'earnings_variability',
        
            # 增长/投资因子
            'growth', 
            'investment_quality',
        
            # 动量与反转因子
            'momentum', 
            'longterm_reversal',
        
            # 流动性/规模因子
            'size', 
            'mid_cap', 
            'liquidity',
        
            # 杠杆因子
            'leverage'
        ]
        
        cols_industry_factor=[
            # 顺周期行业（经济敏感型）
            '煤炭', '石油石化',         # 上游资源
            '有色金属', '钢铁',         # 金属材料
            '基础化工', '建材',         # 中游原材料
            '机械', '建筑',            # 基建与设备制造
            '房地产', '汽车', '家电',   # 下游消费与地产
        
            # 高股息/防御型行业（弱周期）
            '银行', '非银行金融',       # 金融
            '电力及公用事业', '交通运输', # 公共事业
        
            # 消费类行业（需求稳定型）
            '食品饮料', '农林牧渔', '医药',        # 必需消费
            '商贸零售', '消费者服务', '纺织服装', '轻工制造',  # 可选消费
        
            # 科技成长类行业（创新驱动型）
            '计算机', '通信', '电子', '电力设备及新能源', '传媒',
        
            # 其他特殊类别
            '国防军工',               # 政策驱动型
            '综合', '综合金融'         # 多元化业务
        ]
        
        implicit=rqdatac.get_factor_return(st, et, 
                          factors= None, universe='whole_market',
                          method='implicit',industry_mapping='citics_2019', model = 'v2')
        explicit=rqdatac.get_factor_return(st, et, 
                          factors= None, universe='whole_market',
                          method='explicit',industry_mapping='citics_2019', model = 'v2')
        
        ##要用显式因子收益率（多空组合算出来的）+行业因子收益率（显式没有这个），拼起来
        factor_return=pd.concat([explicit[cols_risk_factor],implicit[cols_industry_factor]],axis=1)
        
        ##对因子按天远近加权
        daily_weights=General.sum_normalize([i for i in range(1,len(factor_return)+1)])
        daily_weights = pd.Series(daily_weights, index=factor_return.index)
        factor_return_weighted = factor_return.multiply(daily_weights, axis=0)
        
        
        ########factor_return_map画图
        # factor_return=pd.concat([factor_return[cols_risk_factor],
        #                         factor_return[cols_industry_factor]],axis=1)
        
        # # 创建一个 3 行 2 列的画布
        # fig, axes = plt.subplots(3, 2, figsize=(30, 30))
        # #解决中文或者是负号无法显示的情况
        # mpl.rcParams["font.sans-serif"] = ["SimHei"]
        # mpl.rcParams['axes.unicode_minus'] = False
        # plt.rcParams['figure.dpi'] = 300
        # plt.tight_layout(
        #     pad=10.0,        # 主画布与子图之间的边距
        #     w_pad=10.0,      # 子图之间的水平间距
        #     h_pad=10.0       # 子图之间的垂直间距
        # )
        
        # for id,date in enumerate(factor_return.index):
        #     daily_data=factor_return.loc[date]
        #     risk_part=abs(daily_data[cols_risk_factor]).rank()
        #     industry_part=daily_data[cols_industry_factor].rank()
        #     daily_data = pd.concat([risk_part, industry_part])
            
        #     factor_return_map = []
        #     for risk in cols_risk_factor:
        #         for industry in cols_industry_factor:
        #             factor_return_map.append(abs(daily_data[risk]) + daily_data[industry])
                    
        #     factor_return_map=General.normalize_list(factor_return_map, lower_bound=0, upper_bound=100)
            
        #     factor_return_map=np.array(factor_return_map).reshape(len(cols_risk_factor),len(cols_industry_factor))
        #     factor_return_map=pd.DataFrame(factor_return_map,index=cols_risk_factor,columns=cols_industry_factor)
        #     factor_return_map.index.name=''
        #     factor_return_map.columns.name=''
        #     x,y=divmod(id,2)
        #     sns.heatmap(factor_return_map, annot=False, cmap='coolwarm', ax=axes[x, y])
        #     axes[x, y].set_title(f'{date}')
        
        
        ########factor_return cumsum diagram
        # factor_return=factor_return.cumsum()
        # cols=factor_return.columns
        # Plot.plot_res3(factor_return,'',cols = cols,start_time = factor_return.index[0],
        #                                 end_time=factor_return.index[-1],
        #                                 days = None,
        #                                 maxmin=False)
        
        ########factor_return sharp 筛选策略
        res=factor_return_weighted.describe()
        res=res.T
        res['sharp']=res['mean']/res['std'] ##计算因子收益率sharp
        res['abs_sharp']=abs(res['sharp']) ##非常负的风险因子收益率也是一种市场风格偏向，要看绝对值
        risk_part,industry_part=General.split_dataframe_by_index(res)
        
        
        risk_part=risk_part.sort_values(['abs_sharp'],ascending=False)
        risk_part=risk_part[risk_part['abs_sharp']>0.5] 
        industry_part=industry_part[industry_part['sharp']>0] 
        industry_part=industry_part.sort_values(['sharp'],ascending=False)
        
        
        factor_return_array = []
        for risk in risk_part.index:
            for industry in industry_part.index:
                x1,y1=risk_part['sharp'][risk],risk_part['abs_sharp'][risk]
                x2=industry_part['sharp'][industry]
                factor_return_array.append([risk,industry,x1,y1,x2,y1+x2])
        factor_return_array=pd.DataFrame(factor_return_array,columns = ['risk', 'industry', 'risk_sharp', 'risk_abs_sharp', 'industry_sharp', 'sum_sharp'])
        risk_industry_sharp=factor_return_array.sort_values(['sum_sharp'],ascending=False)
        
        stock_industry_dict={}
        stock_pool_list=[]
        for industry in industry_part.index:
            stocks=rqdatac.get_industry(industry=industry, source='citics_2019', date=None, market='cn')
            stock_pool_list+=stocks
            for stock in stocks:
                stock_industry_dict[stock]=industry
        exposures=rqdatac.get_factor_exposure(stock_pool_list,st,et,factors=None,industry_mapping='citics_2019', model = 'v2')
        
        group = exposures.groupby(level=1)
        stocks_score=[]
        for id,item in group:
            item=item[risk_part.index]
            item_weighted = item.multiply(daily_weights, axis=0)
            exposure=item_weighted.sum()
            industry=stock_industry_dict[id] ##查找该stock对应industry
            info=risk_industry_sharp[risk_industry_sharp['industry']==industry]
            info['exposure']=list(exposure)
            stock_score=sum(info['risk_sharp']*info['exposure']+info['industry_sharp']*abs(info['exposure']))
            stocks_score.append([id,stock_score])
        stocks_score=pd.DataFrame(stocks_score,columns=['id','score'])
        stocks_score=stocks_score.sort_values(['score'],ascending=False)
        stocks_score['name']=stocks_score['id'].apply(lambda id:rqdatac.instruments(id, market='cn').symbol)
        stocks_score=stocks_score.reset_index()
        stocks_score=stocks_score[~stocks_score['name'].str.contains('ST')] ##去掉st的
        stocks_score=stocks_score[stocks_score['score']>0] ##去掉负分的
        
        k=400
        date=et
        select=stocks_score.iloc[:k]
        select['买卖价格']=select['id'].apply(lambda id:rqdatac.get_price(order_book_ids=id, 
                  start_date=date, 
                  end_date=date, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)['close'].iloc[0])
        money=1000e4
        each=money//len(select)
        select['买卖数量']=select['买卖价格'].apply(lambda price:int(each//(price*100)*100))
        select['买卖日期']=date
        select['买卖方向']='买入'
        select['证券代码']=list(select['id'].apply(lambda id:rqdatac.id_convert(id,to='normal')))
        
        res=select[['买卖日期','证券代码', '买卖数量', '买卖价格', '买卖方向']]
        
        acc='acc1'  ##文件名和账户名有关联
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")##文件名和时间有关联
        path=f'./trade_log/{acc}_{now}.xlsx'  
        with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
            res.to_excel(writer, sheet_name='导入数据区', index=False)      
            res3=select[['id','name']]
            res3.to_excel(writer, sheet_name='股票名清单', index=False)              
        
    ####compare_wind_rq 对比wind和rq微盘股
    if args.task=='compare_wind_rq':    
        weights=rqdatac.index_weights(order_book_id='866006.RI', date=args.et)
        weights=weights.reset_index()
        weights['代码']=list(weights['order_book_id'].apply(lambda id:rqdatac.id_convert(id,to='normal')))
        rq=set(weights['代码'])
        wind=pd.read_excel('data/8841431.WI-成分及权重-20250417.xlsx')
        wind=set(wind['代码'])
        
        print(f'交集数量:{len(rq.intersection(wind))}')
        print('rq有，wind没有：')
        print(rq.difference(wind)) 
        print('rq没有，wind有：')
        print(wind.difference(rq)) 
        xx=1
        
    ####make_backtest_file 制作回测所需文件
    if args.task=='make_backtest_file':    
        cols_risk_factor=[
            # 市场风险因子
            'beta', 
            'residual_volatility',
        
            # 价值/基本面因子
            'book_to_price', 
            'dividend_yield', 
            'earnings_yield',
        
            # 质量/盈利因子
            'earnings_quality', 
            'profitability', 
            'earnings_variability',
        
            # 增长/投资因子
            'growth', 
            'investment_quality',
        
            # 动量与反转因子
            'momentum', 
            'longterm_reversal',
        
            # 流动性/规模因子
            'size', 
            'mid_cap', 
            'liquidity',
        
            # 杠杆因子
            'leverage'
        ]
        
        cols_industry_factor=[
            # 顺周期行业（经济敏感型）
            '煤炭', '石油石化',         # 上游资源
            '有色金属', '钢铁',         # 金属材料
            '基础化工', '建材',         # 中游原材料
            '机械', '建筑',            # 基建与设备制造
            '房地产', '汽车', '家电',   # 下游消费与地产
        
            # 高股息/防御型行业（弱周期）
            '银行', '非银行金融',       # 金融
            '电力及公用事业', '交通运输', # 公共事业
        
            # 消费类行业（需求稳定型）
            '食品饮料', '农林牧渔', '医药',        # 必需消费
            '商贸零售', '消费者服务', '纺织服装', '轻工制造',  # 可选消费
        
            # 科技成长类行业（创新驱动型）
            '计算机', '通信', '电子', '电力设备及新能源', '传媒',
        
            # 其他特殊类别
            '国防军工',               # 政策驱动型
            '综合', '综合金融'         # 多元化业务
        ]        
        
        # dates=rqdatac.get_trading_dates(args.st, args.et, market='cn')
        # inputs=[] ##存每天的股票清单
        # for date in tqdm(dates):
        #     st=rqdatac.get_previous_trading_date(date,n=5,market='cn')
        #     et=rqdatac.get_previous_trading_date(date,n=1,market='cn')
       
        #     implicit=rqdatac.get_factor_return(st, et, 
        #                       factors= None, universe='whole_market',
        #                       method='implicit',industry_mapping='citics_2019', model = 'v2')
        #     explicit=rqdatac.get_factor_return(st, et, 
        #                       factors= None, universe='whole_market',
        #                       method='explicit',industry_mapping='citics_2019', model = 'v2')
        
        #     ##要用显式因子收益率（多空组合算出来的）+行业因子收益率（显式没有这个），拼起来
        #     factor_return=pd.concat([explicit[cols_risk_factor],implicit[cols_industry_factor]],axis=1)
            
        #     ##对因子按天远近加权
        #     daily_weights=General.sum_normalize([i for i in range(1,len(factor_return)+1)])
        #     daily_weights = pd.Series(daily_weights, index=factor_return.index)
        #     factor_return_weighted = factor_return.multiply(daily_weights, axis=0)
        
        # ########factor_return_map画图
        # # factor_return=pd.concat([factor_return[cols_risk_factor],
        # #                         factor_return[cols_industry_factor]],axis=1)
        
        # # # 创建一个 3 行 2 列的画布
        # # fig, axes = plt.subplots(3, 2, figsize=(30, 30))
        # # #解决中文或者是负号无法显示的情况
        # # mpl.rcParams["font.sans-serif"] = ["SimHei"]
        # # mpl.rcParams['axes.unicode_minus'] = False
        # # plt.rcParams['figure.dpi'] = 300
        # # plt.tight_layout(
        # #     pad=10.0,        # 主画布与子图之间的边距
        # #     w_pad=10.0,      # 子图之间的水平间距
        # #     h_pad=10.0       # 子图之间的垂直间距
        # # )
        
        # # for id,date in enumerate(factor_return.index):
        # #     daily_data=factor_return.loc[date]
        # #     risk_part=abs(daily_data[cols_risk_factor]).rank()
        # #     industry_part=daily_data[cols_industry_factor].rank()
        # #     daily_data = pd.concat([risk_part, industry_part])
            
        # #     factor_return_map = []
        # #     for risk in cols_risk_factor:
        # #         for industry in cols_industry_factor:
        # #             factor_return_map.append(abs(daily_data[risk]) + daily_data[industry])
                    
        # #     factor_return_map=General.normalize_list(factor_return_map, lower_bound=0, upper_bound=100)
            
        # #     factor_return_map=np.array(factor_return_map).reshape(len(cols_risk_factor),len(cols_industry_factor))
        # #     factor_return_map=pd.DataFrame(factor_return_map,index=cols_risk_factor,columns=cols_industry_factor)
        # #     factor_return_map.index.name=''
        # #     factor_return_map.columns.name=''
        # #     x,y=divmod(id,2)
        # #     sns.heatmap(factor_return_map, annot=False, cmap='coolwarm', ax=axes[x, y])
        # #     axes[x, y].set_title(f'{date}')
        
        
        # ########factor_return cumsum diagram
        # # factor_return=factor_return.cumsum()
        # # cols=factor_return.columns
        # # Plot.plot_res3(factor_return,'',cols = cols,start_time = factor_return.index[0],
        # #                                 end_time=factor_return.index[-1],
        # #                                 days = None,
        # #                                 maxmin=False)
        
            ########factor_return sharp 筛选策略
        #     res=factor_return_weighted.describe()
        #     res=res.T
        #     res['sharp']=res['mean']/res['std'] ##计算因子收益率sharp
        #     res['abs_sharp']=abs(res['sharp']) ##非常负的风险因子收益率也是一种市场风格偏向，要看绝对值
        #     risk_part,industry_part=General.split_dataframe_by_index(res)
            
            
        #     risk_part=risk_part.sort_values(['abs_sharp'],ascending=False)
        #     risk_part=risk_part[risk_part['abs_sharp']>0.5] 
        #     industry_part=industry_part[industry_part['sharp']>0] 
        #     industry_part=industry_part.sort_values(['sharp'],ascending=False)
            
            
        #     factor_return_array = []
        #     for risk in risk_part.index:
        #         for industry in industry_part.index:
        #             x1,y1=risk_part['sharp'][risk],risk_part['abs_sharp'][risk]
        #             x2=industry_part['sharp'][industry]
        #             factor_return_array.append([risk,industry,x1,y1,x2,y1+x2])
        #     factor_return_array=pd.DataFrame(factor_return_array,columns = ['risk', 'industry', 'risk_sharp', 'risk_abs_sharp', 'industry_sharp', 'sum_sharp'])
        #     risk_industry_sharp=factor_return_array.sort_values(['sum_sharp'],ascending=False)
            
        #     stock_industry_dict={}
        #     stock_pool_list=[]
        #     for industry in industry_part.index:
        #         stocks=rqdatac.get_industry(industry=industry, source='citics_2019', date=None, market='cn')
        #         stock_pool_list+=stocks
        #         for stock in stocks:
        #             stock_industry_dict[stock]=industry
        #     exposures=rqdatac.get_factor_exposure(stock_pool_list,st,et,factors=None,industry_mapping='citics_2019', model = 'v2')
            
        #     group = exposures.groupby(level=1)
        #     stocks_score=[]
        #     for id,item in group:
        #         item=item[risk_part.index]
        #         item_weighted = item.multiply(daily_weights, axis=0)
        #         exposure=item_weighted.sum()
        #         industry=stock_industry_dict[id] ##查找该stock对应industry
        #         info=risk_industry_sharp[risk_industry_sharp['industry']==industry]
        #         info['exposure']=list(exposure)
        #         stock_score=sum(info['risk_sharp']*info['exposure']+info['industry_sharp']*abs(info['exposure']))
        #         stocks_score.append([id,stock_score])
        #     stocks_score=pd.DataFrame(stocks_score,columns=['id','score'])
        #     stocks_score=stocks_score.sort_values(['score'],ascending=False)
        #     stocks_score['name']=stocks_score['id'].apply(lambda id:rqdatac.instruments(id, market='cn').symbol)
        #     stocks_score=stocks_score.reset_index()
        #     stocks_score=stocks_score[~stocks_score['name'].str.contains('ST')] ##去掉st的
        #     stocks_score=stocks_score[stocks_score['score']>0] ##去掉负分的
        
        #     select=stocks_score.iloc[:args.k]
        #     k=len(select)
        #     print(k)
        #     if k!=0: ##市场太差，存在筛选是空的
        #         select['TRADE_DT']=date.strftime('%Y%m%d')
        #         select['TARGET_WEIGHT']=1/k
        #         select=select[['TRADE_DT','id','name','TARGET_WEIGHT']]
        #         select.columns=['TRADE_DT','TICKER','NAME','TARGET_WEIGHT']
        #         inputs.append(select)
        #     else:
        #         select=copy.deepcopy(inputs[-1])
        #         # select['TARGET_WEIGHT']=0
        #         select['TRADE_DT']=date.strftime('%Y%m%d')
        #         inputs.append(select)
        #     xx=1
        # inputs=pd.concat(inputs,axis=0)
        # with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
        #     inputs.to_excel(writer, sheet_name='', index=False)                  
                
            
        # select['买卖价格']=select['id'].apply(lambda id:rqdatac.get_price(order_book_ids=id, 
        #           start_date=date, 
        #           end_date=date, 
        #           frequency='1d', 
        #           fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
        #           expect_df=True,time_slice=None)['close'].iloc[0])
        # money=1000e4
        # each=money//len(select)
        # select['买卖数量']=select['买卖价格'].apply(lambda price:int(each//(price*100)*100))
        # select['买卖日期']=date
        # select['买卖方向']='买入'
        # select['证券代码']=list(select['id'].apply(lambda id:rqdatac.id_convert(id,to='normal')))
        
        # res=select[['买卖日期','证券代码', '买卖数量', '买卖价格', '买卖方向']]
        
        # acc='acc1'  ##文件名和账户名有关联
        # now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")##文件名和时间有关联
        # path=f'./trade_log/{acc}_{now}.xlsx'  
        # with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
        #     res.to_excel(writer, sheet_name='导入数据区', index=False)      
        #     res3=select[['id','name']]
        #     res3.to_excel(writer, sheet_name='股票名清单', index=False)      

        inputs=[]
        st=args.st
        et=args.et
        dates=rqdatac.get_trading_dates(st, et, market='cn')
        # is_suspendeds=[]
        for date in tqdm(dates[::args.f]): ##args.f是回测频率
            weights=rqdatac.index_weights(order_book_id='866006.RI', date=date)
            weights=weights.reset_index()
            weights.columns=['TICKER','TARGET_WEIGHT']
            ####组合优化
            new_weight=portfolio_optimize(order_book_ids=list(weights['TICKER']), 
                                date=date, 
                                # objective= MinVariance(), 
                                # objective= MeanVariance(window=252), 
                                objective= MaxSharpeRatio(window=252), 
                                # bnds={'*': (0.002, 0.02)}, 
                                cons=None, 
                                benchmark=None, 
                                cov_model=CovModel.FACTOR_MODEL_DAILY)
            new_weight=new_weight.reset_index()
            new_weight.columns=['TICKER','new_weight']            
            merged = pd.merge(weights, new_weight, on='TICKER')
            weights['TARGET_WEIGHT'] = merged['new_weight']

            weights['TRADE_DT']=date.strftime('%Y%m%d')
            weights['NAME']=[i.symbol for i in rqdatac.instruments(list(weights['TICKER']), market='cn')]
            
            ## 等股
            # weights['close']=list(rqdatac.get_price(order_book_ids=list(weights['TICKER']), 
            #           start_date=date, 
            #           end_date=date, 
            #           frequency='1d', 
            #           fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
            #           expect_df=True,time_slice=None)['close'])
            # weights['TARGET_WEIGHT']=weights['close']/sum(weights['close'])
            
            
            weights=weights[['TRADE_DT','TICKER','NAME','TARGET_WEIGHT']]
            # is_suspendeds+=list(weights['TICKER'])
            inputs.append(weights)
        
        inputs=pd.concat(inputs,axis=0)
        with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
            inputs.to_excel(writer, sheet_name='', index=False)  
            print(f'save to {args.file}')
            
        # is_suspendeds_df=pd.DataFrame(set(is_suspendeds),columns=['id'])
        # is_suspended_df=rqdatac.is_suspended(list(is_suspendeds_df['id']), start_date=st,end_date=et)
        # is_suspended_df.to_csv('config/is_suspended_df.csv') 
        # is_suspended_df=pd.read_csv(r'config/is_suspended_df.csv',index_col=0,parse_dates=True)

    ####make_backtest_file2 制作微盘股+macd回测所需文件
    if args.task=='make_backtest_file2':     
        df2=pd.read_excel('data/米筐微盘股等权日频.xlsx',dtype=str)

        df=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        df.index = df.index.get_level_values(1)
        df['return']=df['close']/df['prev_close']-1
        
        df['DIF'], df['DEA'], df['MACD'] = talib.MACD(df['close'], 
                                                    fastperiod=12, 
                                                    slowperiod=26, 
                                                    signalperiod=9)
        def func1(window):
            # 判断单调性
            threshold=0.04
            buy = (window[1] > window[0]) and (window[2] > window[1]) and (abs(window[1]/window[0]-1)>threshold) and (abs(window[2]/window[1]-1)>threshold)
            sell = (window[0] > window[1]) and (window[1] > window[2]) and (abs(window[1]/window[0]-1)>threshold) and (abs(window[2]/window[1]-1)>threshold)
            return 1 if buy else (-1 if sell else 0)
        
        df['MACD_signal'] = df['MACD'].rolling(window=3, min_periods=1).apply(func1)
        def func3(window):
            return window[1]/window[0]-1      
        df['MACD_pct_change'] = df['MACD'].rolling(window=2, min_periods=1).apply(func3)
        
        df['MACD_flag'] = False ##True代表该天空仓
        # 标记区间的开始和结束
        flag = False  
        for i in range(len(df)):
            index=df.index[i]
            if df.loc[index, 'MACD_signal']==-1 and not flag:
                flag = True
            if df.loc[index, 'MACD_signal']==1 and flag:
                flag = False
            if flag:
                df.loc[index, 'MACD_flag'] = True
        df['MACD_flag']=df['MACD_flag'].shift(1)
        df=df.dropna()
        
        def func2(row):
            if row['MACD_flag']:
                return 0
            else:
                return row['return']
        df['MACD_return']=df.apply(func2, axis=1)
        
        mask = df['MACD_flag'] & ~df['MACD_flag'].shift(1, fill_value=False)
        df['MACD_flag_first_True'] = np.where(df['MACD_flag'], mask, df['MACD_flag'])
        df['MACD_flag_filted'] = df['MACD_flag'] & ~df['MACD_flag_first_True']
        filted_list=list(df[df['MACD_flag_filted']==True].index)
        filted_list=list(map(lambda x:x.strftime("%Y%m%d"),filted_list))
        df3=df2[~df2['TRADE_DT'].isin(filted_list)]
        
        df3_list=list(df3.groupby('TRADE_DT'))
        first_True_list=list(df[df['MACD_flag_first_True']==True].index)
        first_True_list=list(map(lambda x:x.strftime("%Y%m%d"),first_True_list))
        
        df4=[]
        for i in tqdm(range(1,len(df3_list))):
            if df3_list[i][1]['TRADE_DT'].iloc[0] in first_True_list:
                prev=copy.deepcopy(df3_list[i-1][1]) #复制上一个
                prev['TARGET_WEIGHT']=0
                prev['TRADE_DT']=df3_list[i][0]
                df4.append(prev)
            else:
                df4.append(df3_list[i][1])
        df4=pd.concat(df4,axis=0)
        with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
            df4.to_excel(writer, sheet_name='', index=False)  
            print(f'save to {args.file}')        
        xx=a

    ####wpg_macd_pred 用微盘股+macd二阶导判断买卖信号
    if args.task=='wpg_macd_pred':   
        df=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        df.index = df.index.get_level_values(1)
        df['return']=df['close']/df['prev_close']-1
        
        df['DIF'], df['DEA'], df['MACD'] = talib.MACD(df['close'], 
                                                    fastperiod=12, 
                                                    slowperiod=26, 
                                                    signalperiod=9)
        def func1(window):
            # 判断单调性
            threshold=0.04
            buy = (window[1] > window[0]) and (window[2] > window[1]) and (abs(window[1]/window[0]-1)>threshold) and (abs(window[2]/window[1]-1)>threshold)
            sell = (window[0] > window[1]) and (window[1] > window[2]) and (abs(window[1]/window[0]-1)>threshold) and (abs(window[2]/window[1]-1)>threshold)
            return 1 if buy else (-1 if sell else 0)
        
        df['MACD_signal'] = df['MACD'].rolling(window=3, min_periods=1).apply(func1)
        def func3(window):
            return window[1]/window[0]-1      
        df['MACD_pct_change'] = df['MACD'].rolling(window=2, min_periods=1).apply(func3)
        
        df['MACD_flag'] = False ##True代表该天空仓
        # 标记区间的开始和结束
        flag = False  
        for i in range(len(df)):
            index=df.index[i]
            if df.loc[index, 'MACD_signal']==-1 and not flag:
                flag = True
            if df.loc[index, 'MACD_signal']==1 and flag:
                flag = False
            if flag:
                df.loc[index, 'MACD_flag'] = True
        df['MACD_flag']=df['MACD_flag'].shift(1)
        df=df.dropna()
        
        def func2(row):
            if row['MACD_flag']:
                return 0
            else:
                return row['return']
        df['MACD_return']=df.apply(func2, axis=1)
        
        df=df['2017-06-01':]
        print(df.index[0],df.index[-1])
        
        if df.index[-1].strftime('%Y%m%d')!=args.et.strftime('%Y%m%d'):
            print('data is not the lastest!')
        else:
            print(f'save to {args.file}!')
            df[['prev_close', 'volume', 'close', 'total_turnover',
                'return','MACD', 'MACD_signal',
                   'MACD_pct_change', 'MACD_flag']].iloc[-30:].to_csv(args.file)


    ####wpg_market_value_median 查看微盘股市值中位数
    if args.task=='wpg_market_value_median':  
        print('wpg_market_value_median...')
        df=rqdatac.index_weights(order_book_id='866006.RI', date=args.et)
        df=df.reset_index()
        df.columns=['id','weight']
        df['total_share']=list(rqdatac.get_shares(list(df['id']), start_date=args.et, end_date=args.et, fields=None, market='cn', expect_df=True)['total'])

        df['close']=list(rqdatac.get_price(order_book_ids=list(df['id']), 
                  start_date=args.et, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)['close'])  
        df['market_value']=df['close']*df['total_share'] ##市值
        print(df['market_value'].describe())
        xx=1
        
    ####rq_wpg_make_pms_csv 根据米筐微盘成分股等权生成pms目标持仓清单
    if args.task=='rq_wpg_make_pms_csv':  
        print('rq_wpg_make_pms_csv...')
        df=rqdatac.index_weights(order_book_id='866006.RI', date=args.et)
        df=df.reset_index()
        df.columns=['id','weight']
        
        announcement=rqdatac.get_announcement(list(df['id']),'20240101',args.et)
        cc=announcement[announcement['title'].str.contains('立案')]
        cc=cc.reset_index()
        cc=set(cc['order_book_id'])
        df=df[~df['id'].isin(cc)] ##过滤被立案的
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
        # cash = {'证券代码': 'CNY', '买卖数量': '700000', '买卖价格': 1, '买卖方向': '划入'}            
        # cash=pd.DataFrame([cash])
        # res=pd.concat([cash,df2])   
        # res=df2
        # res.insert(0, '买卖日期', '2024-11-26')
        
        acc='绝对收益信用'  ##文件名和账户名有关联
        now = args.et.strftime("%Y-%m-%d")##文件名和时间有关联
        path=f'./PMS_csv/{acc}_{now}.xlsx'  
        with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
            res.to_excel(writer, sheet_name='导入数据区', index=False)   
            print(f'save to {path}')
            res2=df[['id','name']]
            res2.to_excel(writer, sheet_name='股票名清单', index=False)      
        
    ####rq_wpg_adjust_ATX 根据ATX的实时监控.xlsx和米筐的微盘股目标仓位，生成csv，用于ATX 调仓
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
        
        # xx=stock_info[['id','name']]
        # xx.columns=['证券代码','证券名称']
        # df=pd.merge(df,xx,on='证券代码',how='left')
        df=df.reset_index()
        
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

    if args.task=='ATX_to_ATX_adjust':    
        ####ATX_to_ATX_adjust 根据ATX的实时监控.xlsx,生成csv，用于ATX 清仓
        print('ATX_to_ATX_adjust...')

        df=pd.read_excel(args.ATX_pos_file,dtype=str)
        df['证券市场']=df['交易市场'].apply(lambda x:'SZ' if x=='深交所' else 'SH')
        df['证券代码']=df['证券代码']+'.'+df['证券市场']
        df['当前拥股']=df['持仓数量'].apply(int)
        df['目标拥股']=df['持仓数量'].apply(lambda x: int(float(x) * args.ratio))
        df['调整股数']=df['目标拥股']-df['当前拥股']
        df['调整股数']=df['调整股数'].apply(lambda x: round(x / 100) * 100)
        df=df.sort_values('调整股数',ascending=True)
        df=df[df['调整股数']!=0]
        
        df['算法类型']='TWAP'
        df['账户名称']='百榕全天候宏观对冲绝对收益信用'
        df['算法实例']='kf_twap_plus'
        df['证券代码']=df['证券代码']
        df['交易方向']=df['调整股数'].apply(lambda x:'买入' if x>0 else '卖出')
        df['任务数量']=df['调整股数'].apply(abs)
        df['开始时间']=args.st
        df['结束时间']=args.et
        df['涨跌停是否继续执行']='涨停不卖跌停不买'
        df['过期后是否继续执行']='否'
        df['其他参数']=np.nan
        df['交易市场']=np.nan
        
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
        
    if args.task=='ATX_to_PMS_track':    
        ####ATX_to_PMS_track 根据实际ATX 成交查询.xlsx 成交价格，生成PMS单 追踪
        print('ATX_to_PMS_track...')
        
        df2=pd.read_excel(args.ATX_file,dtype=str)
        date=df2['成交日期'].iloc[0].replace('/','-')
        df2['证券市场']=df2['交易市场'].apply(lambda x:'SZ' if x=='深交所' else 'SH')
        df2['证券代码']=df2['证券代码']+'.'+df2['证券市场']
        df2=df2[['证券代码','成交数量','成交价格','交易方向']]
        df2.columns=['证券代码', '买卖数量', '买卖价格', '买卖方向']
        
        # df2['买卖数量']=df2['买卖数量'].apply(lambda x:float(x.replace(',', '')))
        # df2 = df2.groupby('证券代码').agg({'买卖数量': 'sum','买卖价格': 'first','买卖方向': 'first'}).reset_index()
        # df2['买卖数量'] = df2['买卖数量'].apply(lambda x: math.ceil(x / 100) * 100)
        
        # cash = {'证券代码': 'CNY', '买卖数量': '920000', '买卖价格': 1, '买卖方向': '划入'}            
        # cash=pd.DataFrame([cash])
        # res=pd.concat([cash,df2])   
        res=df2
        res.insert(0, '买卖日期', date)
        
        acc='绝对收益信用'  ##文件名和账户名有关联
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")##文件名和时间有关联
        path=f'./PMS_csv/{acc}_{now}.xlsx'  
        with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
            res.to_excel(writer, sheet_name='导入数据区', index=False)   
            print(f'save to {path}')


    ####wpg_adjust_dif 微盘股如果周频调仓，每次调整数量
    if args.task=='wpg_adjust_dif':   
        
        dates=rqdatac.get_trading_dates(args.st, args.et, market='cn')
        res=[] 
        res2=[]
        for date in tqdm(dates):        
            weights=rqdatac.index_weights(order_book_id='866006.RI', date=date)
            weights=weights.reset_index()
            res.append(set(weights['order_book_id']))
        res=res[::5]
        for i in range(1,len(res)):
            res2.append(len(res[i]-res[i-1]))
        print(np.mean(res2))
        xx=1

    ####wpg_drop_study 微盘股大跌研究
    if args.task=='wpg_drop_study':   
        df=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)        
        lag=21
        df['y']=df['close'].shift(-1*lag)/df['close']-1
        xx=1

    ####crowdedness_study 微盘股拥挤度研究
    if args.task=='crowdedness_study':   
        df=pd.read_csv('data/wpg_crowdedness.csv',index_col=0,parse_dates=True)
        from scipy.stats import percentileofscore
        df=df['2020-01-01':]
        df['crowdedness_percent'] = [percentileofscore(df['crowdedness'], v) for v in df['crowdedness']]
        # plt.plot(df['crowdedness'])
        
        lag=99
        df['y']=df['close'].shift(-1*lag)/df['close']-1
        df2=df.dropna()
        df2.to_csv('data/wpg_crowdedness_percent.csv')
        plt.scatter(list(df2['crowdedness_percent']),list(df2['y']))
        plt.xlabel('Crowdedness')
        plt.ylabel(f'未来{lag}天涨幅')
        plt.show()
        # corr=df2['y'].corr(df2['crowdedness_percent'])
        
        
        # for lag in range(1,360):
        #     df['y']=df['close'].shift(-1*lag)/df['close']-1
        #     df2=df.dropna()
        #     corr=df2['y'].corr(df2['crowdedness_percent'])
        #     print(f'{lag}-{corr}')
        xx=1
    ####crowdedness 微盘股拥挤度
    if args.task=='crowdedness':   
        
        #微盘股拥挤度
        dates=rqdatac.get_trading_dates(args.st, args.et, market='cn')
        res=[] ##存每天的微盘股拥挤度
        for date in tqdm(dates):        
            r1=rqdatac.get_price(order_book_ids=args.id1, 
                      start_date=date, 
                      end_date=date, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            r2=rqdatac.get_price(order_book_ids=args.id2, 
                      start_date=date, 
                      end_date=date, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                      start_date=date, 
                      end_date=date, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            
            wpg_turnover=wpg['total_turnover'].iloc[0]
            total_turnover=r1['total_turnover'].iloc[0]+r2['total_turnover'].iloc[0]
            crowdedness=wpg_turnover/total_turnover
            res.append(crowdedness)
            
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)     
        wpg.index = wpg.index.get_level_values(1)
        wpg['crowdedness']=res
        Plot.plot_res(wpg,'',cols = ['close','crowdedness'],start_time = wpg.index[0],
                                        end_time=wpg.index[-1],
                                        days = None,
                                        maxmin=True)
        xx=a

    ####buy_sell逃顶抄底信号实验
    if args.task=='buy_sell':   
        ##微盘股和中证2000
        # wpg=rqdatac.get_price(order_book_ids='866006.RI', 
        #           start_date=args.st, 
        #           end_date=args.et, 
        #           frequency='1d', 
        #           fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
        #           expect_df=True,time_slice=None)
        # zz2000=rqdatac.get_price(order_book_ids='932000.INDX', 
        #           start_date=args.st, 
        #           end_date=args.et, 
        #           frequency='1d', 
        #           fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
        #           expect_df=True,time_slice=None) 
        
        # wpg.index = wpg.index.get_level_values(1)
        # wpg['wpg_return']=wpg['close'].pct_change()
        # zz2000.index = zz2000.index.get_level_values(1)
        # zz2000['zz2000_return']=zz2000['close'].pct_change()
        
        # Metrics.print_metrics(wpg['wpg_return'][1:],wpg.index[1:],0.03)   
        # Metrics.print_metrics(zz2000['zz2000_return'][1:],zz2000.index[1:],0.03)   
        
        
        # df=pd.read_csv('D:/project/quant/rq_app_exp/data/wpg_crowdedness.csv',index_col=0,parse_dates=True)
        df=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        df.index = df.index.get_level_values(1)
        df['return']=df['close']/df['prev_close']-1

        ####M5逃顶抄底法
        if args.method=='compare_rq_wind':
            wind=pd.read_csv('ATX_csv/8841431.WI-行情统计-20250416.csv',index_col=0,parse_dates=True)
            wind = wind.sort_index(ascending=True)
            wind=wind[['涨跌幅(%)']]
            xx=df.join(wind,how='inner')
            xx['涨跌幅(%)']=xx['涨跌幅(%)']/100
            
            Metrics.print_metrics(xx['return'],xx.index,0.03)  
            xx['net']=list(Convert.returns_to_net(xx['return'])) 
            print('\n')
            Metrics.print_metrics(xx['涨跌幅(%)'],xx.index,0.03)   
            xx['wind_net']=list(Convert.returns_to_net(xx['涨跌幅(%)'])) 
            
            Plot.plot_res(xx,'',cols = ["net",
                                        "wind_net",
                                        ],start_time = xx.index[0],
                                              end_time=xx.index[-1],
                                              days = None,
                                              maxmin=False)            
            
            cc=1
        ####M5逃顶抄底法
        if args.method=='M5_1':
            df['M5_volume'] = df['volume'].rolling(window=5).mean()
            
            df['M5'] = df['close'].rolling(window=5).mean()
            df['M10'] = df['close'].rolling(window=10).mean()
            df['M20'] = df['close'].rolling(window=20).mean()
            
            # df['long_arrangement'] = (df['M5'] > df['M10']) & (df['M10'] > df['M20'])
            
            def func1(row):
                above_M5 = row['close'] > row['M5']
                below_M5 = row['close'] < row['M5']
                return 1 if above_M5 else (-1 if below_M5 else 0)
            
            df['M5_signal'] = df.apply(func1,axis=1)
            df['M5_flag'] = False ##True代表该天空仓
            # 标记区间的开始和结束
            flag = False  
            for i in range(len(df)):
                index=df.index[i]
                if df.loc[index, 'M5_signal']==-1 and not flag:
                    flag = True
                if df.loc[index, 'M5_signal']==1 and flag:
                    flag = False
                if flag:
                    df.loc[index, 'M5_flag'] = True
            df['M5_flag']=df['M5_flag'].shift(1)
            df=df.dropna()
            def func2(row):
                if row['M5_flag']:
                    return 0
                else:
                    return row['return']
            df['M5_return']=df.apply(func2, axis=1)
            
            df=df['2017-06-01':]
            print(df.index[0],df.index[-1])
            def check_inconsistency(window):
                return window[0] != window[1]
            res = df['M5_flag'].rolling(window=2).apply(check_inconsistency, raw=True)
            inconsistency_count = res.sum()
            print(f'交易次数：{inconsistency_count}')
            
            Metrics.print_metrics(df['return'],df.index,0.03)  
            df['net']=list(Convert.returns_to_net(df['return'])) 
            print('\n')
            Metrics.print_metrics(df['M5_return'],df.index,0.03)   
            df['M5_net']=list(Convert.returns_to_net(df['M5_return'])) 
            
            Plot.plot_res(df,'',cols = ["net",
                                        "M5_net",
                                        ],start_time = df.index[0],
                                              end_time=df.index[-1],
                                              days = None,
                                              maxmin=False)

        ####M5+volume逃顶抄底法
        if args.method=='M5_2':
            df['M5_volume'] = df['volume'].rolling(window=5).mean()
            
            df['M5'] = df['close'].rolling(window=5).mean()
            df['M10'] = df['close'].rolling(window=10).mean()
            df['M20'] = df['close'].rolling(window=20).mean()
            
            # df['long_arrangement'] = (df['M5'] > df['M10']) & (df['M10'] > df['M20'])
            
            def func1(row):
                # 判断单调性
                above_M5 = (row['close'] > row['M5']) and (row['volume'] > row['M5_volume'])
                below_M5 = (row['close'] < row['M5']) and (row['volume'] > row['M5_volume'])

                return 1 if above_M5 else (-1 if below_M5 else 0)
            
            df['M5_signal'] = df.apply(func1,axis=1)
            df['M5_flag'] = False ##True代表该天空仓
            # 标记区间的开始和结束
            flag = False  
            for i in range(len(df)):
                index=df.index[i]
                if df.loc[index, 'M5_signal']==-1 and not flag:
                    flag = True
                if df.loc[index, 'M5_signal']==1 and flag:
                    flag = False
                if flag:
                    df.loc[index, 'M5_flag'] = True
            df['M5_flag']=df['M5_flag'].shift(1)
            df=df.dropna()
            def func2(row):
                if row['M5_flag']:
                    return 0
                else:
                    return row['return']
            df['M5_return']=df.apply(func2, axis=1)
            
            df=df['2017-06-01':]
            print(df.index[0],df.index[-1])
            def check_inconsistency(window):
                return window[0] != window[1]
            res = df['M5_flag'].rolling(window=2).apply(check_inconsistency, raw=True)
            inconsistency_count = res.sum()
            print(f'交易次数：{inconsistency_count}')
            
            Metrics.print_metrics(df['return'],df.index,0.03)  
            df['net']=list(Convert.returns_to_net(df['return'])) 
            print('\n')
            Metrics.print_metrics(df['M5_return'],df.index,0.03)   
            df['M5_net']=list(Convert.returns_to_net(df['M5_return'])) 
            
            Plot.plot_res(df,'',cols = ["net",
                                        "M5_net",
                                        ],start_time = df.index[0],
                                              end_time=df.index[-1],
                                              days = None,
                                              maxmin=False)

        ####M5+多头排列逃顶抄底法
        if args.method=='M5_3':
            df['M5_volume'] = df['volume'].rolling(window=5).mean()
            
            df['M5'] = df['close'].rolling(window=5).mean()
            df['M10'] = df['close'].rolling(window=10).mean()
            df['M20'] = df['close'].rolling(window=20).mean()
            
            def func1(row):
                # 判断单调性
                buy = (row['M5'] > row['M10']) & (row['M10'] > row['M20'])
                sell = (row['close'] < row['M5']) and (row['volume'] > row['M5_volume'])

                return 1 if buy else (-1 if sell else 0)
            
            df['M5_signal'] = df.apply(func1,axis=1)
            df['M5_flag'] = False ##True代表该天空仓
            # 标记区间的开始和结束
            flag = False  
            for i in range(len(df)):
                index=df.index[i]
                if df.loc[index, 'M5_signal']==-1 and not flag:
                    flag = True
                if df.loc[index, 'M5_signal']==1 and flag:
                    flag = False
                if flag:
                    df.loc[index, 'M5_flag'] = True
            df['M5_flag']=df['M5_flag'].shift(1)
            df=df.dropna()
            def func2(row):
                if row['M5_flag']:
                    return 0
                else:
                    return row['return']
            df['M5_return']=df.apply(func2, axis=1)
            
            df=df['2017-06-01':]
            print(df.index[0],df.index[-1])
            def check_inconsistency(window):
                return window[0] != window[1]
            res = df['M5_flag'].rolling(window=2).apply(check_inconsistency, raw=True)
            inconsistency_count = res.sum()
            print(f'交易次数：{inconsistency_count}')
            
            Metrics.print_metrics(df['return'],df.index,0.03)  
            df['net']=list(Convert.returns_to_net(df['return'])) 
            print('\n')
            Metrics.print_metrics(df['M5_return'],df.index,0.03)   
            df['M5_net']=list(Convert.returns_to_net(df['M5_return'])) 
            
            Plot.plot_res(df,'',cols = ["net",
                                        "M5_net",
                                        ],start_time = df.index[0],
                                              end_time=df.index[-1],
                                              days = None,
                                              maxmin=False)
        
        
        ####macd逃顶抄底法
        if args.method=='MACD_1':
            df['DIF'], df['DEA'], df['MACD'] = talib.MACD(df['close'], 
                                                        fastperiod=12, 
                                                        slowperiod=26, 
                                                        signalperiod=9)
            def func1(window):
                # 判断单调性
                buy = (window[1] > window[0]) and (window[2] > window[1])
                sell = (window[0] > window[1]) and (window[1] > window[2])
                return 1 if buy else (-1 if sell else 0)
            
            df['MACD_signal'] = df['MACD'].rolling(window=3, min_periods=1).apply(func1)
            df['MACD_flag'] = False ##True代表该天空仓
            # 标记区间的开始和结束
            flag = False  
            for i in range(len(df)):
                index=df.index[i]
                if df.loc[index, 'MACD_signal']==-1 and not flag:
                    flag = True
                if df.loc[index, 'MACD_signal']==1 and flag:
                    flag = False
                if flag:
                    df.loc[index, 'MACD_flag'] = True
            df['MACD_flag']=df['MACD_flag'].shift(1)
            df=df.dropna()
            
            def func2(row):
                if row['MACD_flag']:
                    return 0
                else:
                    return row['return']
            df['MACD_return']=df.apply(func2, axis=1)
            
            df=df['2017-06-01':]
            print(df.index[0],df.index[-1])
            def check_inconsistency(window):
                return window[0] != window[1]
            res = df['MACD_flag'].rolling(window=2).apply(check_inconsistency, raw=True)
            inconsistency_count = res.sum()
            print(f'交易次数：{inconsistency_count}')
            
            Metrics.print_metrics(df['return'],df.index,0.03)  
            df['net']=list(Convert.returns_to_net(df['return'])) 
            print('\n')
            Metrics.print_metrics(df['MACD_return'],df.index,0.03)   
            df['MACD_net']=list(Convert.returns_to_net(df['MACD_return'])) 
            
            Plot.plot_res(df,'',cols = ["net",
                                        "MACD_net",
                                        ],start_time = df.index[0],
                                              end_time=df.index[-1],
                                              days = None,
                                              maxmin=False)

        ####macd+斜率控制逃顶抄底法
        if args.method=='MACD_2':
            df['DIF'], df['DEA'], df['MACD'] = talib.MACD(df['close'], 
                                                        fastperiod=12, 
                                                        slowperiod=26, 
                                                        signalperiod=9)
            def func1(window):
                # 判断单调性
                threshold=0.04
                buy = (window[1] > window[0]) and (window[2] > window[1]) and (abs(window[1]/window[0]-1)>threshold) and (abs(window[2]/window[1]-1)>threshold)
                sell = (window[0] > window[1]) and (window[1] > window[2]) and (abs(window[1]/window[0]-1)>threshold) and (abs(window[2]/window[1]-1)>threshold)
                return 1 if buy else (-1 if sell else 0)
            
            df['MACD_signal'] = df['MACD'].rolling(window=3, min_periods=1).apply(func1)
            df['MACD_flag'] = False ##True代表该天空仓
            # 标记区间的开始和结束
            flag = False  
            for i in range(len(df)):
                index=df.index[i]
                if df.loc[index, 'MACD_signal']==-1 and not flag:
                    flag = True
                if df.loc[index, 'MACD_signal']==1 and flag:
                    flag = False
                if flag:
                    df.loc[index, 'MACD_flag'] = True
            df['MACD_flag']=df['MACD_flag'].shift(1)
            df=df.dropna()
            
            def func2(row):
                if row['MACD_flag']:
                    return 0
                else:
                    return row['return']
            df['MACD_return']=df.apply(func2, axis=1)
            
            df=df['2017-06-01':]
            print(df.index[0],df.index[-1])
            def check_inconsistency(window):
                return window[0] != window[1]
            res = df['MACD_flag'].rolling(window=2).apply(check_inconsistency, raw=True)
            inconsistency_count = res.sum()
            print(f'交易次数：{inconsistency_count}')
            
            Metrics.print_metrics(df['return'],df.index,0.03)  
            df['net']=list(Convert.returns_to_net(df['return'])) 
            print('\n')
            Metrics.print_metrics(df['MACD_return'],df.index,0.03)   
            df['MACD_net']=list(Convert.returns_to_net(df['MACD_return'])) 
            
            Plot.plot_res(df,'',cols = ["net",
                                        "MACD_net",
                                        ],start_time = df.index[0],
                                              end_time=df.index[-1],
                                              days = None,
                                              maxmin=False)
        
        ####macd顶分型
        if args.method=='MACD_3':
            df['DIF'], df['DEA'], df['MACD'] = talib.MACD(df['close'], 
                                                        fastperiod=12, 
                                                        slowperiod=26, 
                                                        signalperiod=9)
            def func1(window):
                # 判断单调性
                threshold=0.05
                buy = (window[0] > window[1])  and (window[1] > window[2]) and (window[2] < window[3]) and (window[3] < window[4]) 
                sell = (window[0] < window[1])  and (window[1] < window[2]) and (window[2] > window[3]) and (window[3] > window[4]) 
                return 1 if buy else (-1 if sell else 0)
            
            df['MACD_signal'] = df['MACD'].rolling(window=5, min_periods=1).apply(func1)
            df['MACD_flag'] = False ##True代表该天空仓
            # 标记区间的开始和结束
            flag = False  
            for i in range(len(df)):
                index=df.index[i]
                if df.loc[index, 'MACD_signal']==-1 and not flag:
                    flag = True
                if df.loc[index, 'MACD_signal']==1 and flag:
                    flag = False
                if flag:
                    df.loc[index, 'MACD_flag'] = True
            df['MACD_flag']=df['MACD_flag'].shift(1)
            df=df.dropna()
            
            def func2(row):
                if row['MACD_flag']:
                    return 0
                else:
                    return row['return']
            df['MACD_return']=df.apply(func2, axis=1)
            
            df=df['2017-06-01':]
            print(df.index[0],df.index[-1])
            def check_inconsistency(window):
                return window[0] != window[1]
            res = df['MACD_flag'].rolling(window=2).apply(check_inconsistency, raw=True)
            inconsistency_count = res.sum()
            print(f'交易次数：{inconsistency_count}')
            
            Metrics.print_metrics(df['return'],df.index,0.03)  
            df['net']=list(Convert.returns_to_net(df['return'])) 
            print('\n')
            Metrics.print_metrics(df['MACD_return'],df.index,0.03)   
            df['MACD_net']=list(Convert.returns_to_net(df['MACD_return'])) 
            
            Plot.plot_res(df,'',cols = ["net",
                                        "MACD_net",
                                        ],start_time = df.index[0],
                                              end_time=df.index[-1],
                                              days = None,
                                              maxmin=False)
        ####close顶底分型
        if args.method=='top_bot':
            def func1(window):
                # 通过窗口索引获取所有列数据
                idx = window.index
                window = df.loc[idx, ['high', 'low']]
                high0=window.iloc[0]['high']
                high1=window.iloc[1]['high']
                high2=window.iloc[2]['high']
                low0=window.iloc[0]['low']
                low1=window.iloc[1]['low']
                low2=window.iloc[2]['low']
                
                buy = (high1<high0) and (high1<high2) and (low1<low0) and (low1<low2)
                sell = (high1>high0) and (high1>high2) and (low1>low0) and (low1>low2)
                return 1 if buy else (-1 if sell else 0)
            
            df['close_signal'] = df['high'].rolling(window=3, min_periods=3).apply(func1)
            df['close_flag'] = False ##True代表该天空仓
            # 标记区间的开始和结束
            flag = False  
            for i in range(len(df)):
                index=df.index[i]
                if df.loc[index, 'close_signal']==-1 and not flag:
                    flag = True
                if df.loc[index, 'close_signal']==1 and flag:
                    flag = False
                if flag:
                    df.loc[index, 'close_flag'] = True
            df['close_flag']=df['close_flag'].shift(1)
            df=df.dropna()
            
            def func2(row):
                if row['close_flag']:
                    return 0
                else:
                    return row['return']
            df['close_return']=df.apply(func2, axis=1)
            
            df=df['2017-06-01':]
            print(df.index[0],df.index[-1])
            def check_inconsistency(window):
                return window[0] != window[1]
            res = df['close_flag'].rolling(window=2).apply(check_inconsistency, raw=True)
            inconsistency_count = res.sum()
            print(f'交易次数：{inconsistency_count}')
            
            Metrics.print_metrics(df['return'],df.index,0.03)  
            df['net']=list(Convert.returns_to_net(df['return'])) 
            print('\n')
            Metrics.print_metrics(df['close_return'],df.index,0.03)   
            df['close_net']=list(Convert.returns_to_net(df['close_return'])) 
            
            Plot.plot_res(df,'',cols = ["net",
                                        "close_net",
                                        ],start_time = df.index[0],
                                              end_time=df.index[-1],
                                              days = None,
                                              maxmin=False)
    ####backtest 回测
    if args.task=='backtest':         

        __config__ = {
            "base": {
                "accounts": {
                    "STOCK": 6000000,
                },
                "start_date": args.st,
                "end_date": args.et,
            },
            
        
            # "sys_simulation": {
            #     "price_limit": False
            # },
            


            
            "mod": {
                "sys_analyser": {
                    "plot": True,
                    "benchmark": "866006.RI"
                    # "benchmark": "932000.INDX"
                },
                # 费用模块，该模块的配置项用于调整交易的税费
                "sys_transaction_cost": {
                    # 股票最小手续费，单位元
                    "cn_stock_min_commission": 5,
                    # 佣金倍率（即将废弃）
                    "commission_multiplier": 0.125,
                    # 股票佣金倍率,即在默认的手续费率基础上按该倍数进行调整，股票的默认佣金为万八
                    "stock_commission_multiplier": 1,
                    # 期货佣金倍率,即在默认的手续费率基础上按该倍数进行调整，期货默认佣金因合约而异
                    "futures_commission_multiplier": 1,
                    # 印花倍率，即在默认的印花税基础上按该倍数进行调整，股票默认印花税为万分之五，单边收取
                    "tax_multiplier": 1,
                    # 是否使用回测当时时间点对应的真实印花税率
                    "pit_tax": False,
                },
            }
        }
        
        def read_tables_df():
            # need  pandas version 0.21.0+
            # need xlrd
            d_type = {'NAME': str, 'TARGET_WEIGHT': float, 'TICKER': str, 'TRADE_DT': int}
            columns_name = ["TRADE_DT", "TICKER", "NAME", "TARGET_WEIGHT"]
            df = pd.read_excel(args.file, dtype=d_type)
            if not df.columns.isin(d_type.keys()).all():
                raise TypeError("xlsx文件格式必须有{}四列".format(list(d_type.keys())))
            for date, weight_data in df.groupby("TRADE_DT"):
                if round(weight_data["TARGET_WEIGHT"].sum(), 6) > 1:
                    raise ValueError("权重之和出错，请检查{}日的权重".format(date))
            # 转换为米筐order_book_id
            df['TICKER'] = df['TICKER'].apply(lambda x: rqdatac.id_convert(x) if ".OF" not in x else x)
            return df
        
        
        def on_order_failure(context, event):
            # 拒单时，未成功下单的标的放入第二天下单队列中
            order_book_id = getattr(event, "order_book_id", None) or getattr(event.order, "order_book_id", None)
            context.next_target_queue.append(order_book_id)
        
        
        # 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
        def init(context):
        
            df = read_tables_df()  # 调仓权重文件
            context.target_weight = df
            context.adjust_days = set(context.target_weight.TRADE_DT.to_list())  # 需要调仓的日期
            context.target_queue = []  # 当日需要调仓标的队列
            context.next_target_queue = []  # 次日需要调仓标的队列
            context.current_target_table = dict()  # 当前持仓权重比例
            subscribe_event(EVENT.ORDER_CREATION_REJECT, on_order_failure)
            subscribe_event(EVENT.ORDER_UNSOLICITED_UPDATE, on_order_failure)
        
        
        # before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
        def before_trading(context):
            def dt_2_int_dt(dt):
                return dt.year * 10000 + dt.month * 100 + dt.day
        
            dt = dt_2_int_dt(context.now)
            if dt in context.adjust_days:
                today_df = context.target_weight[context.target_weight.TRADE_DT == dt].set_index("TICKER").sort_values(
                    "TARGET_WEIGHT")
                context.target_queue = today_df.index.to_list()  # 更新需要调仓的队列
                context.current_target_table = today_df["TARGET_WEIGHT"].to_dict()
                context.next_target_queue.clear()
                # 非目标持仓 需要清空
                for i in context.portfolio.positions.keys():
                    if i not in context.target_queue:
                        # 非目标权重持仓 需要清空
                        context.target_queue.insert(0, i)
                    else:
                        # 当前持仓权重大于目标持仓权重 需要优先卖出获得资金
                        equity = context.portfolio.positions[i].long.equity + context.portfolio.positions[i].short.equity
                        total_value = context.portfolio.accounts[instruments(i).account_type].total_value
                        current_percent = equity / total_value
                        if current_percent > context.current_target_table[i]:
                            context.target_queue.remove(i)
                            context.target_queue.insert(0, i)
        
        
        # 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
        def handle_bar(context, bar_dict):
            if context.target_queue:
                context.target_queue = list(set(context.target_queue))
                for _ticker in context.target_queue:
                    # flag=is_suspended_df.loc[context.now.strftime('%Y-%m-%d'),_ticker]
                    # if flag:
                    #     continue
                
                    _target_weight = context.current_target_table.get(_ticker, 0)
                    o = order_target_percent(_ticker, round(_target_weight, 6))
                    if o is None:
                        logger.info("[{}]下单失败，该标将于次日下单".format(_ticker))
                        context.next_target_queue.append(_ticker)
                    else:
                        logger.info("[{}]下单成功，现下占比{}%".format(_ticker, round(_target_weight, 6) * 100))
                # 下单完成 下单失败的的在队列context.next_target_queue中
                context.target_queue.clear()
        
        
        # after_trading函数会在每天交易结束后被调用，当天只会被调用一次
        def after_trading(context):
            if context.next_target_queue:
                context.target_queue += context.next_target_queue
                context.next_target_queue.clear()
            if context.target_queue:
                logger.info("未完成调仓的标的:{}".format(context.target_queue))
                
        df = pd.read_excel(args.file)
        is_suspendeds_df=pd.DataFrame(set(df['TICKER']),columns=['id'])
        is_suspended_df=rqdatac.is_suspended(list(is_suspendeds_df['id']), 
                                             start_date=str(df['TRADE_DT'].iloc[0]),
                                             end_date=str(df['TRADE_DT'].iloc[-1]))
        res=run_func(init=init, before_trading=before_trading, after_trading=after_trading, handle_bar=handle_bar,
                  config=__config__)
        xx=a
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    ####入参
    
    ####常用指数
    #中证2000 '932000.INDX'
    #米筐微盘股 '866006.RI'
    # args.id1='000001.XSHG' #上证
    # args.id2='399106.XSHE' #深证
    
    # args.task='make_config'
    
    # args.task='compare_wind_rq'
    # args.et='20250417'
    
    # args.task='make_backtest_file'
    # args.st='20200101'
    # args.et='20250321'
    # args.f=5
    # args.file=r'data/米筐微盘股macd周频.xlsx'
    
    args.task='wpg_drop_study'
    args.st='20170101'
    args.et='20250321'
    
    # args.task='make_backtest_file2'
    # args.st='20200101'
    # args.et='20250321'
    # args.f=1
    # args.file=r'data/米筐微盘股macd日频.xlsx'
    
    # args.task='backtest'
    # args.st='20200101'
    # args.et='20250319'
    # args.file=r'data/米筐微盘股macd日频.xlsx'
    
    # args.task='buy_sell'
    # args.method='MACD_2'
    # args.id1='000001.XSHG'
    # args.id2='399106.XSHE'
    # args.st='20200101'
    # args.et='20250321'
    
    # args.task='stratgy1'
    
    # args.task='make_backtest_file'
    # args.st='20240101'
    # args.et='20241231'
    # args.k=200
    # args.file=r'data/多因子策略等权.xlsx'
    
    # args.task='exp'
    # args.method='compare_rq_wind'
    # args.id1='000001.XSHG'
    # args.id2='399106.XSHE'
    # args.st='20210101'
    # args.et='20250410'
    
    # args.task='wpg_macd_pred'
    # args.st='20170101'
    # args.et=rqdatac.get_latest_trading_date()
    # args.file='signal/wpg_macd_pred.csv'
    
    # args.task='rq_wpg_make_pms_csv'
    # # args.et=rqdatac.get_latest_trading_date()
    # args.et=pd.to_datetime('20250421')
    # args.money=200e4
    
    # args.task='rq_wpg_adjust_ATX'
    # args.ATX_pos_file='ATX_csv/持仓查询none.xlsx'
    # args.pms_file='PMS_csv/绝对收益信用_2025-04-21.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-04-22_1.csv'
    # args.start_time='20250422T093000000'
    # args.end_time=  '20250422T103000000'  
    # args.account='百榕全天候宏观对冲绝对收益信用'
    # args.account='百榕百里挑一稳健一号信用'
    
    # args.task='ATX_to_PMS_track'
    # args.ATX_file='ATX_csv/成交查询_20250421152047.xlsx'
    
    # args.task='ATX_to_ATX_adjust'
    # args.ATX_pos_file='ATX_csv/持仓查询_20250414151014.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-04-15_1.csv'
    # args.st='20250415T093000000'
    # args.et=  '20250415T103000000'  
    # args.ratio=0
    
    # args.task='crowdedness'
    # args.id1='000001.XSHG'
    # args.id2='399106.XSHE'
    # args.st='20240101'
    # args.et='20250418'
    
    # args.task='wpg_adjust_dif'
    # args.st='20240101'
    # args.et='20250414'
    
    # args.task='wpg_market_value_median'
    # args.et='20250410'
    
    # args.task='crowdedness_study'
    
    main(args)
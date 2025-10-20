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
from scipy import stats

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

        
    ####!!!!!!!!!make_backtest_file1 测试因子打分回测所需文件
    if args.task=='make_backtest_file1':     
        inputs=[]
        st=args.st
        et=args.et
        dates=rqdatac.get_trading_dates(st, et, market='cn')
        len_chosen=[]
        for date in tqdm(dates[::args.f]): 
            date_1=rqdatac.get_previous_trading_date(date,n=1,market='cn')
            date_n=rqdatac.get_previous_trading_date(date,n=args.days,market='cn')
            
            weights=[]
            for id in args.ids:
                part=rqdatac.index_weights(order_book_id=id, date=date_1)
                weights.append(part)
            weights=pd.concat(weights)
            weights=weights.reset_index()
            weights = weights.drop_duplicates(subset=['order_book_id'], keep='last')

            exposure=rqdatac.get_factor_exposure(list(weights['order_book_id']), 
                                               date_1, date_1, factors = list(args.factors.keys()),
                                               industry_mapping='citics_2019', model = 'v2')
            # 计算综合评分（值越高越符合目标）
            exposure['score']=0
            for factor in list(args.factors.keys()):
                exposure['score']+=exposure[factor] *args.factors[factor]
                
            exposure=exposure.sort_values('score',ascending=False)
            chosen=set(exposure.iloc[:args.top_n].index.get_level_values(1))
            weights=weights[weights['order_book_id'].isin(chosen)]
            # number=[]
            # for id in args.ids:   ##成分股占比监测
            #     part=rqdatac.index_weights(order_book_id=id, date=date_1)
            #     n=len(weights[weights['order_book_id'].isin(list(part.index))])
            #     number.append([id,n])
            # print(number)
            weights.columns=['TICKER','TARGET_WEIGHT']
            weights['TRADE_DT']=date.strftime('%Y%m%d')
            weights['NAME']=[i.symbol for i in rqdatac.instruments(list(weights['TICKER']), market='cn')]
            weights=weights[['TRADE_DT','TICKER','NAME','TARGET_WEIGHT']]
            weights['TARGET_WEIGHT']=1/len(weights)
            inputs.append(weights)
        inputs=pd.concat(inputs,axis=0)
        with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
            inputs.to_excel(writer, sheet_name='', index=False)  
            print(f'save to {args.file}')        
        
        x=a
        
    ####make_backtest_file2 测试因子正交化回测所需文件
    if args.task=='make_backtest_file2':     
        import statsmodels.api as sm
        inputs=[]
        st=args.st
        et=args.et
        dates=rqdatac.get_trading_dates(st, et, market='cn')
        len_chosen=[]
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
        for date in tqdm(dates[::args.f]): 
            date_1=rqdatac.get_previous_trading_date(date,n=1,market='cn')
            date_n=rqdatac.get_previous_trading_date(date,n=args.days,market='cn')
            
            weights=[]
            for id in args.ids:
                part=rqdatac.index_weights(order_book_id=id, date=date_1)
                weights.append(part)
            weights=pd.concat(weights)
            weights=weights.reset_index()
            weights = weights.drop_duplicates(subset=['order_book_id'], keep='last')

            n=len(args.factors)
            chosen=set(weights['order_book_id'])
            # if n==1:  ##1000,liq,beta,交集100
            #     top_n=100
            # elif n==2:
            #     top_n=450
            # elif n==3:
            #     top_n=1250
            if n==1:  ##1000,liq,交集300
                top_n=300
            elif n==2:
                top_n=450
            elif n==3:
                top_n=1250
            for factor in args.factors:
                exposure=rqdatac.get_factor_exposure(list(weights['order_book_id']), 
                                                   date_1, date_1, factors = cols_risk_factor,
                                                   industry_mapping='citics_2019', model = 'v2')
                if factor=='size' or factor=='liquidity':
                    sort=exposure.sort_values(factor,ascending=True)
                elif factor=='beta':
                    sort=exposure.sort_values(factor,ascending=False)
                sort=sort.reset_index()
                
                X_list=cols_risk_factor.copy()
                X_list.remove(factor)
                # X_list=['momentum']
                X_list=['beta','liquidity','momentum']
                X = sort[X_list]
                X = sm.add_constant(X)  # 添加截距项
                y = sort[factor]  # 因变量：Size因子暴露
                # 线性回归：Y = α + β1*BP + β2*Momentum + β3*Volatility + ε
                model = sm.OLS(y, X).fit()
                sort['residual'] = model.resid  # 残差ε：纯Size暴露（越小越好）
                sort = sort.sort_values('residual')  # 残差越小，纯Size暴露越显著
                sort3=sort.head(100)
                chosen2=set(sort['order_book_id'].iloc[:top_n])
                chosen=chosen & chosen2
            print(len(chosen))
            weights=weights[weights['order_book_id'].isin(chosen)]
            
            weights.columns=['TICKER','TARGET_WEIGHT']
            weights['TRADE_DT']=date.strftime('%Y%m%d')
            weights['NAME']=[i.symbol for i in rqdatac.instruments(list(weights['TICKER']), market='cn')]
            weights=weights[['TRADE_DT','TICKER','NAME','TARGET_WEIGHT']]
            weights['TARGET_WEIGHT']=1/len(weights)
            inputs.append(weights)
        inputs=pd.concat(inputs,axis=0)
        with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
            inputs.to_excel(writer, sheet_name='', index=False)  
            print(f'save to {args.file}')        
        
        x=a
        
    ####make_backtest_file3 自制因子
    if args.task=='make_backtest_file3':     
        inputs=[]
        st=args.st
        et=args.et
        dates=rqdatac.get_trading_dates(st, et, market='cn')
        len_chosen=[]
        for date in tqdm(dates[::args.f]): 
            date_1=rqdatac.get_previous_trading_date(date,n=1,market='cn')
            date_n=rqdatac.get_previous_trading_date(date,n=args.days,market='cn')
            
            weights=[]
            for id in args.ids:
                part=rqdatac.index_weights(order_book_id=id, date=date_1)
                weights.append(part)
            weights=pd.concat(weights)
            weights=weights.reset_index()
            weights = weights.drop_duplicates(subset=['order_book_id'], keep='last')

            n=len(args.factors)
            chosen=set(weights['order_book_id'])
            if n==1:  ##1000+2000,size,liq,beta,交集100
                top_n=100
            elif n==2:
                top_n=800
            elif n==3:
                top_n=1250
            # if n==1:  ##1000+2000,size,liq,beta,交集200
            #     top_n=200
            # elif n==2:
            #     top_n=1000
            # elif n==3:
            #     top_n=1450
            # if n==1:  ##1000+2000,size,liq,beta,交集300
            #     top_n=300
            # elif n==2:
            #     top_n=1200
            # elif n==3:
            #     top_n=1550
            
            
            ##beta复现
            date_1=rqdatac.get_previous_trading_date(date,n=1,market='cn')
            date_252=rqdatac.get_previous_trading_date(date,n=252,market='cn')
            wpg=rqdatac.get_price(order_book_ids=list(weights['order_book_id']), 
                      start_date=date_252, 
                      end_date=date_1, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            hs300=rqdatac.get_price(order_book_ids='000300.XSHG', 
                      start_date=date_252, 
                      end_date=date_1, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            
               
            hs300['return']=hs300['close']/hs300['close'].shift(1)-1
            hs300=hs300.bfill()
     
             # wpg['return']=wpg['close']/wpg['prev_close']-1
             # hs300['return']=hs300['close']/hs300['prev_close']-1
     
            hs300.index = hs300.index.get_level_values(1)
     
            import statsmodels.formula.api as smf
            def func(df,hs300):
                df.index = df.index.get_level_values(1)
                df['return']=df['close']/df['close'].shift(1)-1
                df=df.bfill()
                res=df[['return']].join(hs300[['return']], lsuffix='_wpg', rsuffix='_hs300')
                
                # 公式接口：y ~ x（因变量~自变量，默认添加截距项）
                model_ols = smf.ols(formula='return_wpg ~ return_hs300', data=res)
                result_ols = model_ols.fit()
                beta=result_ols.params.iloc[-1]
                
                # res=General.generate_decay_weight(res)
                # model_wls  = smf.wls(formula='return_wpg ~ return_hs300', data=res, weights=res['weight'])
                # result_wls = model_wls.fit()
                # beta=result_wls.params.iloc[-1]
                
                return beta
            
            df=wpg.groupby(level=0).apply(func,hs300)
            df=df.sort_values(ascending=False)
            
            
            
            chosen2=set(df.index[:top_n])
            
            # ##size复现
            # shares=rqdatac.get_shares(list(weights['order_book_id']), start_date=date, end_date=date, fields=None, market='cn', expect_df=True)['total_a'] 
            # shares.index = shares.index.get_level_values(0)

            # close=rqdatac.get_price(order_book_ids=list(weights['order_book_id']), 
            #           start_date=date, 
            #           end_date=date, 
            #           frequency='1d', 
            #           fields=None, adjust_type='none', skip_suspended =False, market='cn',  ##要不复权的价格！！！
            #           expect_df=True,time_slice=None)['close']
            # close.index = close.index.get_level_values(0)
            
            # df=weights.set_index('order_book_id')
            
            # shares = shares.to_frame(name='total_share') 
            # close = close.to_frame(name='close') 
            # df=df.join(shares).join(close)
            
            # df['market_value']=df['close']*df['total_share'] ##总市值
            # df=df.sort_values('market_value',ascending=True)
            # chosen2=set(df.index[:top_n])
            
            ##liquidity复现
            # turnover=rqdatac.get_turnover_rate(list(weights['order_book_id']), 
            #                                    start_date=date, end_date=date, 
            #                                    fields=None,expect_df=True)
            # turnover.index = turnover.index.get_level_values(0)
            # turnover = turnover[~(turnover == 0).all(axis=1)]
            # turnover['liquidity']=0.35*turnover['week']+0.35*turnover['month']+0.3*turnover['year']
            # turnover=turnover.sort_values('liquidity',ascending=True)
            # chosen2=set(turnover.index[:top_n])
            
            
            chosen=chosen & chosen2
                
            weights=weights[weights['order_book_id'].isin(chosen)]
            
            weights.columns=['TICKER','TARGET_WEIGHT']
            weights['TRADE_DT']=date.strftime('%Y%m%d')
            weights['NAME']=[i.symbol for i in rqdatac.instruments(list(weights['TICKER']), market='cn')]
            weights=weights[['TRADE_DT','TICKER','NAME','TARGET_WEIGHT']]
            weights['TARGET_WEIGHT']=1/len(weights)
            inputs.append(weights)
        inputs=pd.concat(inputs,axis=0)
        with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
            inputs.to_excel(writer, sheet_name='', index=False)  
            print(f'save to {args.file}')        
        
        x=a

        
        
    ####make_backtest_file4 制作微盘股+macd回测所需文件
    if args.task=='make_backtest_file4':     
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
        
    ####make_backtest_file5 制作微盘股1、4、12月空仓，用红利股替代的回测所需文件
    if args.task=='make_backtest_file5':     
        inputs=[]
        st=args.st
        et=args.et
        dates=rqdatac.get_trading_dates(st, et, market='cn')
        hlg_month=[1,4,12]
        for date in tqdm(dates): 
            if date.month in hlg_month:
                print(date.month,'红利')
                weights=rqdatac.index_weights(order_book_id='000922.XSHG', date=date)
            else:
                print(date.month,'微盘股')
                weights=rqdatac.index_weights(order_book_id='866006.RI', date=date)
            weights=weights.reset_index()
            weights.columns=['TICKER','TARGET_WEIGHT']
            weights['TRADE_DT']=date.strftime('%Y%m%d')
            weights['NAME']=[i.symbol for i in rqdatac.instruments(list(weights['TICKER']), market='cn')]
            weights=weights[['TRADE_DT','TICKER','NAME','TARGET_WEIGHT']]
            inputs.append(weights)
        inputs=pd.concat(inputs,axis=0)
        with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
            inputs.to_excel(writer, sheet_name='', index=False)  
            print(f'save to {args.file}')       
                 
            
        
    ####make_backtest_file6 制作1000中考虑均线趋势的size、beta、liquid的回测所需文件
    if args.task=='make_backtest_file6':     
        inputs=[]
        st=args.st
        et=args.et
        dates=rqdatac.get_trading_dates(st, et, market='cn')
        len_chosen=[]
        factors=['size','liquidity']
        # factors=['size','liquidity','beta']
        
        for date in tqdm(dates[::args.f]): 
            date_1=rqdatac.get_previous_trading_date(date,n=1,market='cn')
            date_20=rqdatac.get_previous_trading_date(date,n=60,market='cn')
            
            factor_return=rqdatac.get_factor_return(date_20, date_1, 
                              factors= factors, universe=args.universe,
                              method='implicit',industry_mapping='citics_2019', model = 'v2')
            factor_chosen=[] ##通过因子收益率均线来筛选强势因子
            for factor in factors:
                factor_return[f'{factor}_M10']=factor_return[f'{factor}'].rolling(30).mean()
                factor_return[f'{factor}_M20']=factor_return[f'{factor}'].rolling(60).mean()
                if factor=='size' or factor=='liquidity':
                    if factor_return[f'{factor}_M10'].iloc[-1]<factor_return[f'{factor}_M20'].iloc[-1]:
                        factor_chosen.append(factor)
                if factor=='beta':
                    if factor_return[f'{factor}_M10'].iloc[-1]>factor_return[f'{factor}_M20'].iloc[-1]:
                        factor_chosen.append(factor)
                        
            weights=rqdatac.index_weights(order_book_id='000852.XSHG', date=date_1)
            weights=weights.reset_index()
            
            chosen=set(weights['order_book_id'])
            n=len(factor_chosen)
            if n==1:
                top_n=300
            elif n==2:
                top_n=600
            elif n==3:
                top_n=650
            for factor in factor_chosen:
                exposure=rqdatac.get_factor_exposure(list(weights['order_book_id']), 
                                                   date_1, date_1, factors = factor,
                                                   industry_mapping='citics_2019', model = 'v2')
                if factor=='size' or factor=='liquidity':
                    sort=exposure.sort_values(factor,ascending=True)
                elif factor=='beta':
                    sort=exposure.sort_values(factor,ascending=False)
                sort=sort.reset_index()
                chosen2=set(sort['order_book_id'].iloc[:top_n])
                chosen=chosen & chosen2
            
            weights=weights[weights['order_book_id'].isin(chosen)]
            
            weights.columns=['TICKER','TARGET_WEIGHT']
            weights['TRADE_DT']=date.strftime('%Y%m%d')
            weights['NAME']=[i.symbol for i in rqdatac.instruments(list(weights['TICKER']), market='cn')]
            weights=weights[['TRADE_DT','TICKER','NAME','TARGET_WEIGHT']]
            weights['TARGET_WEIGHT']=1/len(weights)
            inputs.append(weights)
        inputs=pd.concat(inputs,axis=0)
        with pd.ExcelWriter(args.file, engine='xlsxwriter') as writer:
            inputs.to_excel(writer, sheet_name='', index=False)  
            print(f'save to {args.file}')        
        
        x=a

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
        starts=[]
        ends=[]
        for i in range(len(df)):
            index=df.index[i]
            if df.loc[index, 'MACD_signal']==-1 and not flag:
                flag = True
                starts.append(index)
            if df.loc[index, 'MACD_signal']==1 and flag:
                flag = False
                ends.append(index)
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

        lag=3
        df['y']=df['close'].shift(-1*lag)/df['close']-1
        df=df.dropna()
        
        # df=df['2017-06-01':]
        print(df.index[0],df.index[-1])
        
        #找出首次buy 和 sell 信号
        def func4(window):
            if window[0]==0 and window[1]==1:
                return 1
            elif window[0]==0 and window[1]==-1:
                return -1
            else:
                return 0
        df['MACD_first_signal'] = df['MACD_signal'].rolling(window=2, min_periods=2).apply(func4)
        
        
        gaps=list(zip(starts,ends))
        res=[]
        for i in gaps:
            if i[0].year in [2025]:
                print(i[0].year)
                # continue
                close0=df.loc[i[0], 'close']
                close1=df.loc[i[1], 'close']
                r=close1/close0-1
                res.append(r)
        res=pd.DataFrame(res,columns=['return'])
        ratio=len(res[res['return']<0])/len(res)
        print(ratio)
        xx=1
        # df1=df[df['MACD_first_signal']==1]
        # plt.scatter(df1['MACD'],df1['y'])
        # plt.show()
        
        # df2=df[df['MACD_first_signal']==-1]
        # plt.scatter(df2['MACD'],df2['y'])
        # plt.show()
        
        # if df.index[-1].strftime('%Y%m%d')!=args.et.strftime('%Y%m%d'):
        #     print('data is not the lastest!')
        # else:
        #     print(f'save to {args.file}!')
        #     df[['prev_close', 'volume', 'close', 'total_turnover',
        #         'return','MACD', 'MACD_signal',
        #            'MACD_pct_change', 'MACD_flag']].iloc[-30:].to_csv(args.file)


    ####compare_rq_cty_beta_factor 对比自己复现和米筐的beta因子排序是否一致
    if args.task=='compare_rq_cty_beta_factor':  
        print('compare_rq_cty_beta_factor...')
        
        df=rqdatac.index_weights(order_book_id='866006.RI', date=args.et)
        
        date_1=rqdatac.get_previous_trading_date(args.et,n=1,market='cn')
        date_252=rqdatac.get_previous_trading_date(args.et,n=252,market='cn')
        wpg=rqdatac.get_price(order_book_ids=list(df.index), 
                  start_date=date_252, 
                  end_date=date_1, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        hs300=rqdatac.get_price(order_book_ids='000300.XSHG', 
                  start_date=date_252, 
                  end_date=date_1, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        

        hs300['return']=hs300['close']/hs300['close'].shift(1)-1
        hs300=hs300.bfill()
        
        # wpg['return']=wpg['close']/wpg['prev_close']-1
        # hs300['return']=hs300['close']/hs300['prev_close']-1
        
        hs300.index = hs300.index.get_level_values(1)
        
        import statsmodels.formula.api as smf
        def func(df,hs300):
            df.index = df.index.get_level_values(1)
            df['return']=df['close']/df['close'].shift(1)-1
            df=df.bfill()
            res=df[['return']].join(hs300[['return']], lsuffix='_wpg', rsuffix='_hs300')
            
            # 公式接口：y ~ x（因变量~自变量，默认添加截距项）
            # model_ols = smf.ols(formula='return_wpg ~ return_hs300', data=res)
            # result_ols = model_ols.fit()
            # beta=result_ols.params.iloc[-1]
            
            res=General.generate_decay_weight(res)
            model_wls  = smf.wls(formula='return_wpg ~ return_hs300', data=res, weights=res['weight'])
            result_wls = model_wls.fit()
            beta=result_wls.params.iloc[-1]
            
            return beta
        
        res=wpg.groupby(level=0).apply(func,hs300)
        res=res.sort_values(ascending=False)
        
        
        factor=rqdatac.get_factor_exposure(list(df.index), 
                                           args.et, args.et, factors = ['beta'],
                                           industry_mapping='citics_2019', model = 'v2')
        
        beta_sort=factor.sort_values('beta',ascending=False)
        beta_sort.index = beta_sort.index.get_level_values(1)
        print(res.index==beta_sort.index)
        
        n=100
        s1=set(res.index[:n])
        s2=set(beta_sort.index[:n])
        print(len(s1 & s2))
        xx=1
        
    ####compare_rq_cty_size_factor 对比自己复现和米筐的size因子排序是否一致
    if args.task=='compare_rq_cty_size_factor':  
        print('compare_rq_cty_size_factor...')
        
        df=rqdatac.index_weights(order_book_id='866006.RI', date=args.et)
        
        shares=rqdatac.get_shares(list(df.index), start_date=args.et, end_date=args.et, fields=None, market='cn', expect_df=True)['total_a'] ##A股总股本
        shares.index = shares.index.get_level_values(0)

        close=rqdatac.get_price(order_book_ids=list(df.index), 
                  start_date=args.et, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='none', skip_suspended =False, market='cn',  ##要不复权的价格！！！
                  expect_df=True,time_slice=None)['close']
        close.index = close.index.get_level_values(0)
        
        df=pd.DataFrame([df,close,shares]).T
        df.columns=['weight','close','total_share']
        df['market_value']=df['close']*df['total_share'] ##总市值
        df=df.sort_values('market_value',ascending=True)
        
        factor=rqdatac.get_factor_exposure(list(df.index), 
                                           args.et, args.et, factors = ['size'],
                                           industry_mapping='citics_2019', model = 'v2')
        
        size_sort=factor.sort_values('size',ascending=True)
        size_sort.index = size_sort.index.get_level_values(1)
        print(df.index==size_sort.index)
        xx=1
        
    ####compare_rq_cty_liquidity_factor 对比自己复现和米筐的liquidity因子排序是否一致
    if args.task=='compare_rq_cty_liquidity_factor':  
        print('compare_rq_cty_liquidity_factor...')
        
        df=rqdatac.index_weights(order_book_id='866006.RI', date=args.et)
        
        # date_1=rqdatac.get_previous_trading_date(args.et,n=1,market='cn')
        # date_252=rqdatac.get_previous_trading_date(args.et,n=252,market='cn')
        # turnover=rqdatac.get_turnover_rate(list(df.index), 
        #                                    start_date=date_252, end_date=date_1, 
        #                                    fields=None,expect_df=True)        
        # turnover=turnover.reset_index()
        # def func(df):
        #     STOM=np.log(df['today'][-21:].sum())
        #     STOQ=np.log(df['today'][-21*3:].sum())
        #     STOA=np.log(df['today'][-21*12:].sum())
            
        #     # # 步骤1：计算衰减系数λ
        #     # lambda_ = np.log(2) / 63 
        #     # # 步骤2：生成原始指数衰减权重（最近数据权重最高）
        #     # n = len(df)  # 252
        #     # t = np.arange(n)  # t = [0, 1, ..., 251]，对应t-1（i从0开始）
        #     # raw_weights = np.exp(-lambda_ * t)  # 原始权重：最近一天(t=0)权重=exp(0)=1，最早一天(t=251)权重最小
        #     # # 步骤3：归一化权重（确保总和为1）
        #     # normalized_weights = raw_weights / raw_weights.sum()
        #     # # 将权重添加到DataFrame（注意：权重与原始数据顺序一致，即最早数据对应最小权重，最近数据对应最大权重）
        #     # df['weight'] = normalized_weights[::-1]
        #     # ATVR=(df['today']*df['weight']).mean()
        #     return 0.35*STOM+0.35*STOQ+0.3*STOA
        
        # turnover=turnover.groupby('order_book_id').apply(func)
        # turnover=turnover.sort_values(ascending=True)
        
        turnover=rqdatac.get_turnover_rate(list(df.index), 
                                           start_date=args.et, end_date=args.et, 
                                           fields=None,expect_df=True)
        turnover.index = turnover.index.get_level_values(0)
        turnover['liquidity']=0.35*turnover['week']+0.35*turnover['month']+0.3*turnover['year']
        turnover=turnover.sort_values('liquidity',ascending=True)
        
        factor=rqdatac.get_factor_exposure(list(df.index), 
                                           args.et, args.et, factors = ['liquidity'],
                                           industry_mapping='citics_2019', model = 'v2')
        
        liquidity_sort=factor.sort_values('liquidity',ascending=True)
        liquidity_sort.index = liquidity_sort.index.get_level_values(1)
        print(turnover.index==liquidity_sort.index)
        
        n=100
        s1=set(turnover.index[:n])
        s2=set(liquidity_sort.index[:n])
        print(len(s1 & s2))
        # s3=set(df.index[:n])
        # print(len(s1 & s3))
        xx=1
        
        
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
        
    ####rq_wpg_make_pms_csv 根据米筐微盘成分股等权生成pms目标持仓清单,liq+size
    if args.task=='rq_wpg_make_pms_csv':  
        print('rq_wpg_make_pms_csv...')
        df=rqdatac.index_weights(order_book_id=args.id, date=args.et) ##中证500
        df=df.reset_index()
        df.columns=['id','weight']
        
        ##过滤被立案的
        announcement=rqdatac.get_announcement(list(df['id']),'20240101',args.et)
        cc=announcement[announcement['title'].str.contains('立案')]
        cc=cc.reset_index()
        cc=set(cc['order_book_id'])
        df=df[~df['id'].isin(cc)] 
        
        
        #取size和liquidity top300的交集
        factor=rqdatac.get_factor_exposure(list(df['id']), 
                                           args.et, args.et, factors = ['size','liquidity'],
                                           industry_mapping='citics_2019', model = 'v2')
        
        size_sort=factor.sort_values('size',ascending=True)
        size_sort=size_sort.reset_index()
        liquidity_sort=factor.sort_values('liquidity',ascending=True)
        liquidity_sort=liquidity_sort.reset_index()
        
        size_chosen=set(size_sort['order_book_id'].iloc[:args.n])
        liquidity_chosen=set(liquidity_sort['order_book_id'].iloc[:args.n])
        chosen=list(size_chosen.intersection(liquidity_chosen))
        print('len_chosen:',len(chosen))
            
            
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
        # cash = {'证券代码': 'CNY', '买卖数量': '700000', '买卖价格': 1, '买卖方向': '划入'}            
        # cash=pd.DataFrame([cash])
        # res=pd.concat([cash,df2])   
        # res=df2
        # res.insert(0, '买卖日期', '2024-11-26')
        
        acc='共同target'  ##文件名和账户名有关联
        now = args.et.strftime("%Y-%m-%d")##文件名和时间有关联
        path=f'./PMS_csv/{acc}_{now}.xlsx'  
        with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
            res.to_excel(writer, sheet_name='导入数据区', index=False)   
            print(f'save to {path}')
            res2=df[['id','name']]
            res2.to_excel(writer, sheet_name='股票名清单', index=False)      
            
    ####rq_wpg_make_pms_csv2 根据米筐微盘成分股等权生成pms目标持仓清单,liq
    if args.task=='rq_wpg_make_pms_csv2':  
        print('rq_wpg_make_pms_csv2...')
        df=rqdatac.index_weights(order_book_id=args.id, date=args.et) ##中证500
        df=df.reset_index()
        df.columns=['id','weight']
        
        ##过滤被立案的
        announcement=rqdatac.get_announcement(list(df['id']),'20240101',args.et)
        cc=announcement[announcement['title'].str.contains('立案')]
        cc=cc.reset_index()
        cc=set(cc['order_book_id'])
        df=df[~df['id'].isin(cc)] 
        
        
        #取size和liquidity top300的交集
        factor=rqdatac.get_factor_exposure(list(df['id']), 
                                           args.et, args.et, factors = ['size','liquidity'],
                                           industry_mapping='citics_2019', model = 'v2')
        
        liquidity_sort=factor.sort_values('liquidity',ascending=True)
        liquidity_sort=liquidity_sort.reset_index()
        
        chosen=list(liquidity_sort['order_book_id'].iloc[:args.n])
            
            
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
        # cash = {'证券代码': 'CNY', '买卖数量': '700000', '买卖价格': 1, '买卖方向': '划入'}            
        # cash=pd.DataFrame([cash])
        # res=pd.concat([cash,df2])   
        # res=df2
        # res.insert(0, '买卖日期', '2024-11-26')
        
        acc='共同target'  ##文件名和账户名有关联
        now = args.et.strftime("%Y-%m-%d")##文件名和时间有关联
        path=f'./PMS_csv/{acc}_{now}.xlsx'  
        with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
            res.to_excel(writer, sheet_name='导入数据区', index=False)   
            print(f'save to {path}')
            res2=df[['id','name']]
            res2.to_excel(writer, sheet_name='股票名清单', index=False)  
            
            
    ####!!!!!!!rq_wpg_make_pms_csv3 根据任意池子+任意因子等权生成pms目标持仓清单
    if args.task=='rq_wpg_make_pms_csv3':  
        print('rq_wpg_make_pms_csv3...')
        
        weights=[]
        for id in args.ids:
            part=rqdatac.index_weights(order_book_id=id, date=args.et)
            weights.append(part)
        weights=pd.concat(weights)
        weights=weights.reset_index()
        weights = weights.drop_duplicates(subset=['order_book_id'], keep='last')
        
        ##过滤被立案的
        announcement=rqdatac.get_announcement(list(weights['order_book_id']),'20240101',args.et)
        cc=announcement[announcement['title'].str.contains('立案')]
        cc=cc.reset_index()
        cc=set(cc['order_book_id'])
        weights=weights[~weights['order_book_id'].isin(cc)]        
        
        n=len(args.factors)
        chosen=set(weights['order_book_id'])
        if n==1:  ##1000+2000,size,liq,beta,交集100
            top_n=100
        elif n==2:
            top_n=850
        elif n==3:
            top_n=1250
        for factor in args.factors:
            exposure=rqdatac.get_factor_exposure(list(weights['order_book_id']), 
                                               args.et, args.et, factors = factor,
                                               industry_mapping='citics_2019', model = 'v2')
            if factor=='size' or factor=='liquidity':
                sort=exposure.sort_values(factor,ascending=True)
            elif factor=='beta':
                sort=exposure.sort_values(factor,ascending=False)
            sort=sort.reset_index()
            chosen2=set(sort['order_book_id'].iloc[:top_n])
            chosen=chosen & chosen2
            
        df=weights[weights['order_book_id'].isin(chosen)]
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
        
        buy_money=res[res['买卖方向']=='买入']
        if len(buy_money)!=0:
            buy_money=sum(buy_money['买卖价格']*buy_money['买卖数量'])
        else:
            buy_money=0
            
        sell_money=res[res['买卖方向']=='卖出']
        if len(sell_money)!=0:
            sell_money=sum(buy_money['买卖价格']*buy_money['买卖数量'])
        else:
            sell_money=0        
        print('总数:',money)
        print('买入:',buy_money)
        print('卖出:',sell_money)
        
        acc='共同target'  ##文件名和账户名有关联
        now = args.et.strftime("%Y-%m-%d")##文件名和时间有关联
        path=f'./PMS_csv/{acc}_{now}.xlsx'  
        with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
            res.to_excel(writer, sheet_name='导入数据区', index=False)   
            print(f'save to {path}')
            res2=df[['id','name']]
            res2.to_excel(writer, sheet_name='股票名清单', index=False)      
        
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
        
        # xx=stock_info[['id','name']]
        # xx.columns=['证券代码','证券名称']
        # df=pd.merge(df,xx,on='证券代码',how='left')
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
        df['账户名称']=args.account
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
        
    if args.task=='ATX_to_ATX_adjust2':    
        ####ATX_to_ATX_adjust2 根据ATX的实时监控.xlsx,生成csv，用于ATX 将正盈利的票清仓
        print('ATX_to_ATX_adjust2...')

        df=pd.read_excel(args.ATX_pos_file,dtype=str)
        
        df=df[df['持仓盈亏'].apply(lambda x: float(x))>0] ##正盈利的清仓
        
        df['证券市场']=df['交易市场'].apply(lambda x:'SZ' if x=='深交所' else 'SH')
        df['证券代码']=df['证券代码']+'.'+df['证券市场']
        df['当前拥股']=df['持仓数量'].apply(int)
        df['目标拥股']=df['持仓数量'].apply(lambda x: int(float(x) * args.ratio))
        df['调整股数']=df['目标拥股']-df['当前拥股']
        # df['调整股数']=df['调整股数'].apply(lambda x: round(x / 100) * 100)
        df=df.sort_values('调整股数',ascending=True)
        df=df[df['调整股数']!=0]
        
        df['算法类型']='TWAP'
        df['账户名称']=args.account
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
        
    if args.task=='ATX_to_ATX_adjust3':    
        ####ATX_to_ATX_adjust3 根据ATX的实时监控.xlsx,生成csv，用于ATX 将正盈利的票按百分比减仓
        print('ATX_to_ATX_adjust3...')

        df=pd.read_excel(args.ATX_pos_file,dtype=str)
        
        df['证券市场']=df['交易市场'].apply(lambda x:'SZ' if x=='深交所' else 'SH')
        df['证券代码']=df['证券代码']+'.'+df['证券市场']
        df['当前拥股']=df['持仓数量'].apply(int)
        df['最新价']=df['最新价'].apply(float)
        df['持仓盈亏']=df['持仓盈亏'].apply(float)
        df['持仓市值']=df['最新价']*df['当前拥股']
        total_market=df['持仓市值'].sum()
        reduce=total_market*(1-args.ratio)
        df=df[df['持仓盈亏'].apply(lambda x: float(x))>0] ##正盈利的清仓
        df=df.sort_values(['持仓盈亏'],ascending=False)
        
        s=0
        res=[]
        for index, row in df.iterrows():
            s+=row.持仓市值
            res.append(index)
            if s>reduce:
                break
        df=df.loc[res]
        df['目标拥股']=0
        df['调整股数']=df['目标拥股']-df['当前拥股']
        # df['调整股数']=df['调整股数'].apply(lambda x: round(x / 100) * 100)
        df=df.sort_values('调整股数',ascending=True)
        df=df[df['调整股数']!=0]
        
        df['算法类型']='TWAP'
        df['账户名称']=args.account
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
        print(f'save to {args.ATX_file}')
        
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


    ####wpg_hlg 微盘-红利涨幅
    if args.task=='wpg_hlg':   
        dates=rqdatac.get_trading_dates(start_date='20200101', end_date='20250101')
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=dates[0], 
                  end_date=dates[-1], 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)  
        hlg=rqdatac.get_price(order_book_ids='000922.XSHG', 
                  start_date=dates[0], 
                  end_date=dates[-1], 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)  

        wpg.index = wpg.index.get_level_values(1)
        hlg.index = hlg.index.get_level_values(1)
        T=30
        hlg['hlg_moment']=(hlg['close'] - hlg['close'].shift(T)) / hlg['close'].shift(T)
        wpg['wpg_moment']=(wpg['close'] - wpg['close'].shift(T)) / wpg['close'].shift(T)
        
        res=wpg[['close','wpg_moment']].join(hlg['hlg_moment'])
        res['crowdedness']=res['wpg_moment']-res['hlg_moment']

        res=res.dropna()

        from scipy.stats import percentileofscore
        res['crowdedness_percent'] = [percentileofscore(res['crowdedness'], v) for v in res['crowdedness']]
        
        res=res.sort_values(['crowdedness_percent'],ascending=True)
        res.to_csv('data/wpg-hlg_percent.csv')
        x=a
        
        
        
        
        
        
        
    ####wpg_maxdrop_study 微盘股最大回撤研究
    if args.task=='wpg_maxdrop_study':   
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None) 
        wpg.index = wpg.index.get_level_values(1)
        
        def calculate_max_drawdown(window_series):
            """计算单个窗口内的最大回撤"""
            arr = window_series.values  # 转换为numpy数组加速计算
            # 从后往前计算每个位置的最小值（包括当前位置）
            min_from_i = np.minimum.accumulate(arr[::-1])[::-1]
            # 计算每个位置的回撤：(后续最小值 - 当前值) / 当前值
            drawdowns = (min_from_i - arr) / arr
            return drawdowns.min()  # 取最小的回撤（最大跌幅）
        def calculate_max_gain(window_series):
            """计算单个窗口内的最大涨幅"""
            arr = window_series.values  # 转换为numpy数组加速计算
            # 从后往前计算每个位置的最大值（包括当前位置）
            max_from_i = np.maximum.accumulate(arr[::-1])[::-1]
            # 计算每个位置的涨幅：(后续最大值 - 当前值) / 当前值
            gains = (max_from_i - arr) / arr
            return gains.max()  # 取最大的涨幅（最大收益率）
        # 计算每个90天窗口的最大回撤
        wpg['max_drawdown'] = wpg['close'].rolling(window=60, min_periods=60).apply(calculate_max_drawdown)
        wpg['max_drawdown'].hist()
        wpg['max_gain'] = wpg['close'].rolling(window=60, min_periods=60).apply(calculate_max_gain)
        wpg['max_gain'].hist()
        
        xx=a
        
    ####wpg_maxdrop_study2 微盘股和中证1000回撤对应关系研究
    if args.task=='wpg_maxdrop_study2':   
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None) 
        wpg.index = wpg.index.get_level_values(1)
        
        zz1000=rqdatac.get_price(order_book_ids='000852.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None) 
        zz1000.index = zz1000.index.get_level_values(1)
        
        # 计算每个90天窗口的最大回撤
        def func(window):
            return  window[-1]/window[0]-1
            
        wpg['wpg_drawdown'] = wpg['close'].rolling(window=21, min_periods=21).apply(func)
        zz1000['zz1000_drawdown'] = zz1000['close'].rolling(window=21, min_periods=21).apply(func)
        
        res=wpg[['wpg_drawdown']].join(zz1000[['zz1000_drawdown']])
        xx=res[res['wpg_drawdown']<-0.1]
        xx=a
        
    ####wpg_zz1000_beta 微盘股和中证1000主连对冲的beta计算
    if args.task=='wpg_zz1000_beta':   
        zz1000=rqdatac.futures.get_dominant_price(underlying_symbols='IM'
                                                 ,start_date=args.st,end_date=args.et,
                                                 frequency='1d',fields=None,
                                                 adjust_type='none', 
                                                 adjust_method='prev_close_ratio')
        zz1000.index = zz1000.index.get_level_values(1)
        
        # wpg=rqdatac.get_price(order_book_ids='866006.RI', 
        wpg=rqdatac.get_price(order_book_ids='932000.INDX', 
        # wpg=rqdatac.get_price(order_book_ids='000852.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None) 
        wpg.index = wpg.index.get_level_values(1)
        
        zz1000['return']=zz1000['close']/zz1000['prev_close']-1
        wpg['return']=wpg['close']/wpg['prev_close']-1
        res=wpg[['return']].join(zz1000[['return']], lsuffix='_wpg', rsuffix='_zz1000')
        res['delta']=res['return_wpg']/res['return_zz1000']
        
        res=res[res['return_wpg']<0]
        
        res['delta'].plot()
        print(res['delta'].describe())
        print(len(res[res['delta']<0])/len(res))
        xx=1
    ####yz_return_study 预增指数每月收益研究
    if args.task=='yz_return_study':   
        
        df=pd.read_excel('data/8841090.WI.xlsx',index_col='日期',parse_dates=True)
        
        df['year'] = df.index.year
        df['month'] = df.index.month
        df.rename(columns={'收盘价(元)': 'close'}, inplace=True)
        
        group1=list(df.groupby(['year', 'month']))
        ys=[]
        for i,item in enumerate(group1):
            if i==0:
                y=group1[i][1]['close'].iloc[-1] / group1[i][1]['close'].iloc[0] - 1
            else:
                y=group1[i][1]['close'].iloc[-1] / group1[i-1][1]['close'].iloc[-1] - 1
            ys.append([item[0],y])
            
        df1 = pd.DataFrame(ys, columns=['Date', 'Return'])
        df1['Year'], df1['Month'] = zip(*df1['Date'])
        df1 = df1.drop(columns=['Date'])
        df1 = df1.pivot(index='Year', columns='Month', values='Return')                
        df1['年度累计'] = (df1.iloc[:, :12]+1).prod(axis=1)-1
        # monthly_returns = df1.applymap(lambda x: "{:.2%}".format(x))
        monthly_returns=df1
        monthly_returns.index.name='月度收益'
        res_wpg=monthly_returns.describe()
        
        xx=monthly_returns.iloc[:,:12]
        xx = np.where(xx > 0, 1, -1) #正收益为1，负收益-1
        xx = xx.flatten().tolist()
        xx=xx[:-6] ##去掉最后位nan的6个月
        
        def count_consecutive(xx): ##统计连续1 和-1 的长度
            if not xx:  # 处理空列表
                return {'-1': [], '1': []}
            
            result = {'-1': [], '1': []}
            current_val = xx[0]  # 初始当前值为第一个元素
            current_length = 1   # 初始长度为1
            
            for num in xx[1:]:   # 从第二个元素开始遍历
                if num == current_val:
                    current_length += 1  # 与当前值相同，长度+1
                else:
                    # 遇到不同值，保存当前长度到对应键的列表
                    result[str(current_val)].append(current_length)
                    current_val = num    # 更新当前值为新值
                    current_length = 1   # 重置长度为1
            
            # 遍历结束后，保存最后一个连续段的长度
            result[str(current_val)].append(current_length)
            return result
        
        dic=count_consecutive(xx)
        
        

        xx=a
        
    ####wpg_hlg_return_study 微盘股和红利股每月收益研究
    if args.task=='wpg_hlg_return_study':   
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None) 
        wpg.index = wpg.index.get_level_values(1)
        wpg['year'] = wpg.index.year
        wpg['month'] = wpg.index.month
        
        group1=list(wpg.groupby(['year', 'month']))
        ys=[]
        for i,item in enumerate(group1):
            if i==0:
                y=group1[i][1]['close'].iloc[-1] / group1[i][1]['close'].iloc[0] - 1
            else:
                y=group1[i][1]['close'].iloc[-1] / group1[i-1][1]['close'].iloc[-1] - 1
            ys.append([item[0],y])
            
        df1 = pd.DataFrame(ys, columns=['Date', 'Return'])
        df1['Year'], df1['Month'] = zip(*df1['Date'])
        df1 = df1.drop(columns=['Date'])
        df1 = df1.pivot(index='Year', columns='Month', values='Return')                
        df1['年度累计'] = (df1.iloc[:, :12]+1).prod(axis=1)-1
        # monthly_returns = df1.applymap(lambda x: "{:.2%}".format(x))
        monthly_returns=df1
        monthly_returns.index.name='月度收益'
        res_wpg=monthly_returns.describe()
        
        xx=monthly_returns.iloc[:,:12]
        xx = np.where(xx > 0, 1, -1) #正收益为1，负收益-1
        xx = xx.flatten().tolist()
        xx=xx[:-6] ##去掉最后位nan的6个月
        
        def count_consecutive(xx): ##统计连续1 和-1 的长度
            if not xx:  # 处理空列表
                return {'-1': [], '1': []}
            
            result = {'-1': [], '1': []}
            current_val = xx[0]  # 初始当前值为第一个元素
            current_length = 1   # 初始长度为1
            
            for num in xx[1:]:   # 从第二个元素开始遍历
                if num == current_val:
                    current_length += 1  # 与当前值相同，长度+1
                else:
                    # 遇到不同值，保存当前长度到对应键的列表
                    result[str(current_val)].append(current_length)
                    current_val = num    # 更新当前值为新值
                    current_length = 1   # 重置长度为1
            
            # 遍历结束后，保存最后一个连续段的长度
            result[str(current_val)].append(current_length)
            return result
        
        dic=count_consecutive(xx)
        
        
        hlg=rqdatac.get_price(order_book_ids='000922.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None) 
        hlg.index = hlg.index.get_level_values(1)
        hlg['year'] = hlg.index.year
        hlg['month'] = hlg.index.month
        
        group1=list(hlg.groupby(['year', 'month']))
        ys=[]
        for i,item in enumerate(group1):
            if i==0:
                y=group1[i][1]['close'].iloc[-1] / group1[i][1]['close'].iloc[0] - 1
            else:
                y=group1[i][1]['close'].iloc[-1] / group1[i-1][1]['close'].iloc[-1] - 1
            ys.append([item[0],y])
            
        df1 = pd.DataFrame(ys, columns=['Date', 'Return'])
        df1['Year'], df1['Month'] = zip(*df1['Date'])
        df1 = df1.drop(columns=['Date'])
        df1 = df1.pivot(index='Year', columns='Month', values='Return')                
        df1['年度累计'] = (df1.iloc[:, :12]+1).prod(axis=1)-1
        # monthly_returns = df1.applymap(lambda x: "{:.2%}".format(x))
        monthly_returns2=df1
        monthly_returns2.index.name='月度收益'
        res_hlg=monthly_returns2.describe()        

        xx=a
        
        
    ####hlg_dividend 红利股息率监测
    if args.task=='hlg_dividend':   
        dates=rqdatac.get_trading_dates(args.st, args.et, market='cn')
        result=[]
        for date in tqdm(dates):    
            riskfree=rqdatac.get_yield_curve(start_date=date, end_date=date)
            
            df=rqdatac.index_weights(order_book_id='000922.XSHG', date=date) ##中证红利
            df=df.reset_index()
            
            dividend=rqdatac.get_factor(list(df['order_book_id']), 'dividend_yield_ttm', date,date)
            dividend=dividend.reset_index()
            
            res=dividend['dividend_yield_ttm'].mean()
            
            df2=rqdatac.index_weights(order_book_id='399986.XSHE', date=date) ##中证银行
            df2=df2.reset_index()
            
            dividend2=rqdatac.get_factor(list(df2['order_book_id']), 'dividend_yield_ttm', date,date)
            dividend2=dividend2.reset_index()
            
            res2=dividend2['dividend_yield_ttm'].mean()
            
            result.append([res,res2])
        result=pd.DataFrame(result,columns=['zzhl','zzyh'])
        
        xx=a
    ####factor_study 因子研究
    if args.task=='factor_study':   
        factors=rqdatac.get_factor_return(args.st, args.et, 
                          factors= None, universe='whole_market',
                          method='implicit',industry_mapping='citics_2019', model = 'v2')
        
        for factor in factors.columns:
            res=factors[[factor]].cumsum()
            res[factor].plot(title=factor)
            plt.show()
        
        xx=1
    ####!!!!!!!factor_study2 因子研究2-看特定时间段因子表现
    if args.task=='factor_study2':   
        factors=rqdatac.get_factor_return(args.st, args.et, 
                          factors= None, universe=args.universe,
                          method='implicit',industry_mapping='citics_2019', model = 'v2')
        
        cols=['beta', 'book_to_price', 'dividend_yield',
               'earnings_quality', 'earnings_variability', 'earnings_yield', 'growth',
               'investment_quality', 'leverage', 'liquidity', 'longterm_reversal',
               'mid_cap', 'momentum', 'profitability', 'residual_volatility', 'size']
        
        
        # plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        
        factors_cumsum = factors.cumsum()
        n = len(cols)
        
        # ---------------------- 绘制多子图（按列分布）----------------------
        # 计算子图网格尺寸（4列布局，自动计算行数）
        cols_per_row = 5
        rows = math.ceil(n / cols_per_row)
        figsize = (30, 5 * rows)  # 宽度15，高度按行数动态调整
        
        # 创建子图网格
        fig, axes = plt.subplots(rows, cols_per_row, figsize=figsize)
        axes = axes.flatten()  # 展平为一维数组，方便循环
        
        # 遍历列名绘制子图
        for i, col in enumerate(cols):
            ax = axes[i]
            ax.plot(factors_cumsum[col])
            ax.plot(factors_cumsum[col].rolling(window=10).mean(),label='M10')
            ax.plot(factors_cumsum[col].rolling(window=20).mean(),label='M20')
            ax.legend()
            ax.set_title(f"{col}", fontsize=12, pad=10)
            ax.grid(alpha=0.3)
        
        # 隐藏未使用的子图（当列数不足网格总数时）
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout(pad=3)  # 调整子图间距
        plt.show()
        
        # ---------------------- 绘制汇总大图（含所有折线）----------------------
        plt.figure(figsize=(12, 6))  # 大图尺寸
        for col in cols:
            plt.plot(factors_cumsum[col], label=col, linewidth=2)
        
        plt.title("所有指标累积和趋势对比", fontsize=14, pad=20)
        plt.xlabel("时间", fontsize=12)
        plt.ylabel("累积值", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
        plt.grid(alpha=0.3)
        plt.tight_layout()  # 防止标签被截断
        plt.show()
        
        xx=1
        
    ####factor_study3 单因子研究
    if args.task=='factor_study3':   
        
        df=rqdatac.index_weights(order_book_id='866006.RI', date=args.st)
        df=df.reset_index()
        df.columns=['id','weight']
        ids=list(df['id'])
        factors=rqdatac.get_factor_exposure(ids,args.st,args.et,factors=[args.factor],
                                              industry_mapping='citics_2019', model = 'v2')
        factors=factors[args.factor]
        # factors.index.set_names(['date', 'asset'], inplace=True) 
        # factors = pd.to_datetime(factors.index.get_level_values(0))
        
        
        df2=rqdatac.get_price(order_book_ids=ids, 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields='close', adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)      
        df2=df2.reset_index()
        prices = df2.pivot(index='date', columns='order_book_id', values='close')
        

        factor_data = get_clean_factor_and_forward_returns(
                                                 factors,
                                                 prices,
                                                 groupby=None,
                                                 binning_by_group=False,
                                                 quantiles=5,
                                                 bins=None,
                                                 periods=(1, 5, 10),
                                                 filter_zscore=20,
                                                 groupby_labels=None,
                                                 max_loss=0.35,
                                                 zero_aware=False,
                                                 cumulative_returns=True)
        
        # alphalens.tears.create_information_tear_sheet(factor_data, group_neutral, by_group, set_context=False)
        alphalens.tears.create_full_tear_sheet(factor_data)
        xx=1
        
    ####backtest1 500,1000,2000,wpg,hlg综合回测
    if args.task=='backtest1':   
        zz500=rqdatac.get_price(order_book_ids='000905.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        zz1000=rqdatac.get_price(order_book_ids='000852.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        zz2000=rqdatac.get_price(order_book_ids='932000.INDX', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        hlg=rqdatac.get_price(order_book_ids='000922.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)  

        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        
        zz500.index = zz500.index.get_level_values(1)
        zz1000.index = zz1000.index.get_level_values(1)
        zz2000.index = zz2000.index.get_level_values(1)
        hlg.index = hlg.index.get_level_values(1)
        wpg.index = wpg.index.get_level_values(1)
        
        res=pd.DataFrame([zz500['close'],zz1000['close'],
                          zz2000['close'],hlg['close'],wpg['close']]).T
        res.columns=['zz500','zz1000','zz2000','hlg','wpg']
        for col in res.columns:
            res[col]=res[col]/res[col].iloc[0]
        res['portfolio']=(res['zz500']+res['zz1000']+res['zz2000']+res['hlg']+res['wpg'])/5
        
        res['portfolio_return']=res['portfolio']/res['portfolio'].shift(1)-1
        res=res.dropna()
        Metrics.print_metrics(res['portfolio_return'],res.index,0.017) 
        Plot.plot_res(res,'',cols = ['zz500','zz1000','zz2000','hlg','wpg','portfolio'],
                                        start_time = res.index[0],
                                        end_time=res.index[-1],
                                        days = None,
                                        maxmin=False)
        xx=1
    ####factor_study4 每日因子IC，IR跟踪
    if args.task=='factor_study4':   
        
        df=rqdatac.index_weights(order_book_id='000985.XSHG', date=args.st)
        # df=rqdatac.index_weights(order_book_id='866006.RI', date=args.st)
        df=df.reset_index()
        df.columns=['id','weight']
        ids=list(df['id'])
        
        for factor in args.factor[:1]:
            factors=rqdatac.get_factor_exposure(ids,args.st,args.et,factors=[factor],
                                                  industry_mapping='citics_2019', model = 'v2')
            factors=factors[factor]
            
            
            df2=rqdatac.get_price(order_book_ids=ids, 
                      start_date=args.st, 
                      end_date=args.et, 
                      frequency='1d', 
                      fields='close', adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)      
            df2=df2.reset_index()
            prices = df2.pivot(index='date', columns='order_book_id', values='close')
            
            
            res = []
            for i in tqdm(range(len(prices))):
                df = prices.iloc[i:i+args.window, :]  
                if len(df)<args.window:
                    break
                st=df.index[0]
                et=df.index[-1]
                factor_data = get_clean_factor_and_forward_returns(
                                                         factors.loc[df.index],
                                                         df,
                                                         groupby=None,
                                                         binning_by_group=False,
                                                         quantiles=5,
                                                         bins=None,
                                                         periods=(1, 5, 10),
                                                         filter_zscore=20,
                                                         groupby_labels=None,
                                                         max_loss=0.35,
                                                         zero_aware=False,
                                                         cumulative_returns=True)
                
                ic_data = alphalens.performance.factor_information_coefficient(factor_data)
                ic_data=ic_data[['5D']]
                mean=ic_data.mean()[0]
                std=ic_data.std()[0]
                ir=mean/std
                skew=stats.skew(ic_data)[0]
                kurtosis=stats.kurtosis(ic_data)[0]
                
                res.append([st,et,mean,std,ir,skew,kurtosis])
                
                # ic_summary_table = pd.DataFrame()
                # ic_summary_table["IC Mean"] = ic_data.mean()
                # ic_summary_table["IC Std."] = ic_data.std()
                # ic_summary_table["Risk-Adjusted IC"] = ic_data.mean() / ic_data.std()
                # t_stat, p_value = stats.ttest_1samp(ic_data, 0)
                # ic_summary_table["t-stat(IC)"] = t_stat
                # ic_summary_table["p-value(IC)"] = p_value
                # ic_summary_table["IC Skew"] = stats.skew(ic_data)
                # ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)
                xx=1
        
        # alphalens.tears.create_full_tear_sheet(factor_data)
        res=pd.DataFrame(res,columns=['st','et','mean','std','ir','skew','kurtosis'])
        res=res.set_index('et')
        xx=a
    
    ####1000_2000_beta_study 用IM来做2000多头的对冲，计算beta
    if args.task=='1000_2000_beta_study':   
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)      
        zz1000=rqdatac.futures.get_dominant_price(underlying_symbols='IM'
                                                 ,start_date=args.st,end_date=args.et,
                                                 frequency='1d',fields=None,
                                                 adjust_type='none', 
                                                 adjust_method='prev_close_ratio')
        zz2000=rqdatac.get_price(order_book_ids='932000.INDX', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)      
        wpg.index = wpg.index.get_level_values(1)
        zz1000.index = zz1000.index.get_level_values(1)
        zz2000.index = zz2000.index.get_level_values(1)
        
        wpg['return']=wpg['close']/wpg['prev_close']-1
        zz1000['return']=zz1000['close']/zz1000['prev_close']-1
        zz2000['return']=zz2000['close']/zz2000['prev_close']-1
        
        df=pd.concat([wpg['return'],zz1000['return'],zz2000['return']],axis=1)
        df.columns=['wpg_ret','zz1000_ret','zz2000_ret']
        
        zz1000_std=df['zz1000_ret'].std()
        zz2000_std=df['zz2000_ret'].std()
        
        corr=df['zz1000_ret'].corr(df['zz2000_ret'])
        # 准备自变量（X）和因变量（y）
        X = df[['zz1000_ret']]  # 需为二维数组（DataFrame格式）
        y = df['zz2000_ret']
        
        # 训练线性回归模型
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # 提取回归结果
        alpha_hat = model.intercept_  # 估计的α（截距）
        beta_hat = model.coef_[0]     # 估计的β（斜率）
        
        hedge_ratio=corr*zz2000_std/zz1000_std
        plt.figure(figsize=(10, 6), dpi=100)
        
        print(f'beta:{beta_hat}')
        print(f'corr:{corr}')
        print(f'zz1000_std:{zz1000_std}')
        print(f'zz2000_std:{zz2000_std}')
        print(f'hedge_ratio:{hedge_ratio}')
        # 绘制散点图（收益率关系）
        plt.scatter(
            df['zz1000_ret'], 
            df['zz2000_ret'], 
            color='lightblue', 
            alpha=0.6, 
            label='每日收益率散点'
        )
        
        # 绘制回归线
        x_range = np.linspace(df['zz1000_ret'].min(), df['zz1000_ret'].max(), 100)
        y_pred = model.predict(pd.DataFrame(x_range, columns=['zz1000_ret']))
        plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'回归线: y = {alpha_hat:.6f} + {beta_hat:.4f}x')
        
        # 添加标题和标签
        plt.title('中证2000与中证1000日收益率线性回归分析', fontsize=14)
        plt.xlabel('中证1000日收益率', fontsize=12)
        plt.ylabel('中证2000日收益率', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(linestyle='--', alpha=0.7)
        
        # 显示图形
        plt.show()
        xx=1
        
    ####zzqz_study 中证全指研究
    if args.task=='zzqz_study':   
        df=rqdatac.index_weights(order_book_id='000985.XSHG', date=args.st)
        df=df.reset_index()
        df.columns=['id','weight']
        df2=rqdatac.get_price(order_book_ids=list(df['id']), 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields='close', adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)      
        df2=df2.reset_index()
        df2=df2.set_index('date')
        money=df2['close'].sum()*100
        res=df2.groupby(level=0)['close'].sum()*100
        res=res.to_frame()
        res['return']=res['close']/res['close'].shift(1)-1
        res=res.fillna(0)
        res['net']=list(Convert.returns_to_net(res['return'])) 
        
        print('自己编制等股全A')
        Metrics.print_metrics(res['return'],res.index,0.03) 
        
        zzqz=rqdatac.get_price(order_book_ids='000985.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields='close', adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)           
        zzqz.index = zzqz.index.get_level_values(1)
        res['zzqz_net']=zzqz['close']/zzqz['close'].iloc[0]
        
        res['net2']=res['close']/res['close'].iloc[0]
        
        Plot.plot_res(res,'',cols = ['net','zzqz_net'],start_time = res.index[0],
                                        end_time=res.index[-1],
                                        days = None,
                                        maxmin=False)
        
    ####yz_study wind预增指数研究
    if args.task=='yz_study':   
        yz=pd.read_excel('data/8841090.WI.xlsx')
        yz=yz.set_index('日期')
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=yz.index[0], 
                  end_date=yz.index[-1], 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)      
        wpg.index = wpg.index.get_level_values(1)
        
        res=yz[['收盘价(元)']].join(wpg[['close']])
        res.columns=['yz_close','wpg_close']
        res['yz_net']=res['yz_close']/res['yz_close'].iloc[0]
        res['wpg_net']=res['wpg_close']/res['wpg_close'].iloc[0]
        
        Plot.plot_res(res,'',cols = ['yz_net','wpg_net'],start_time = res.index[0],
                                        end_time=res.index[-1],
                                        days = None,
                                        maxmin=False)
        Metrics.print_metrics(yz['涨跌幅'],yz.index,0.017) 
        xx=1
        
    ####basis_study IM基差研究
    if args.task=='basis_study':   


        # months = ['2301','2302','2304','2305','2307','2308','2310','2311',
        #           '2401','2402','2404','2405','2407','2408','2410','2411',
        #           '2501','2502','2504','2505','2507','2508']
        months = ['2303','2306','2309','2312','2403','2406','2409','2412','2503','2506','2509']
        basiss=[]
        for month in months:
            basis=rqdatac.futures.get_basis(order_book_ids=f'IM{month}', 
                                            start_date=20220101,end_date=20251001,
                                            fields=None,frequency='1d')
            basis.index = basis.index.get_level_values(1)
            basiss.append(basis)
        
        
        fig, axes = plt.subplots(nrows=len(basiss), ncols=1, figsize=(10, 16), sharex=True)  # sharex=True共享x轴
        res=[]
        for i, ax in enumerate(axes):  # axes是子图坐标轴列表，共4个元素（0-3对应4行）
            xx=basiss[i]['basis'].reset_index(drop=True)
            res.append(xx)
            xx.plot(ax=ax)  # 指定ax参数将图表绘制到对应子图
            ax.set_title(f'Subplot {i+1}')  # 设置子图标题（可选）
            ax.grid(linestyle='--', alpha=0.7)  # 添加网格线（可选）
         
        plt.tight_layout()
        plt.show()
        
        cc=pd.concat(res,axis=1)
        cc=cc.ffill()
        cc['mean'] = cc.mean(axis=1)
        cc['mean'].plot()
        x=1
        # zz1000=rqdatac.get_price(order_book_ids='000852.XSHG', 
        #           start_date=basis.index[0], 
        #           end_date=basis.index[-1], 
        #           frequency='1d', 
        #           fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
        #           expect_df=True,time_slice=None) 
        # zz1000.index = zz1000.index.get_level_values(1)
        
        # res=basis[['close']].join(zz1000[['close']], lsuffix='_basis', rsuffix='_zz1000')
        # res['basis_ratio']=(res['close_basis']-res['close_zz1000'])/res['close_zz1000']
        # res['basis_ratio'].hist()
        # print(res['basis_ratio'].describe())
        # res['basis_ratio'].plot()
        # xx=1
        
        
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
        df=df.sort_values(['y'],ascending=True)
        focus=df[df['y']<-0.1]
        focus=focus.sort_values(['date'],ascending=True)
        focus=focus.reset_index()
        focus['year_month'] = focus['date'].dt.to_period('M').apply(str)
        group=focus.groupby('year_month')
        res=[]
        for date,v in group:
            res.append([date,v['y'].mean()])
        res=pd.DataFrame(res,columns=['date','21day_return'])
        xx=1

    ####crowdedness_study1 拥挤度门限测试
    if args.task=='crowdedness_study1':   
        
        r1=rqdatac.get_price(order_book_ids=args.id1, 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        r2=rqdatac.get_price(order_book_ids=args.id2, 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        
        r1.index = r1.index.get_level_values(1)
        r2.index = r2.index.get_level_values(1)
        wpg.index = wpg.index.get_level_values(1)
        
        res=wpg['total_turnover']/(r1['total_turnover']+r2['total_turnover'])
        res.name='crowdedness'
        res=res.to_frame()
        
        days=args.w*252
        res=res.iloc[-days:]
        
        from scipy.stats import percentileofscore
        res['crowdedness_percent'] = [percentileofscore(res['crowdedness'], v) for v in res['crowdedness']]
        res=res.join(wpg[['close']])
        
        res['y']=res['close'].shift(-1*args.t)/res['close']-1
        res=res.dropna()
        
        res=res.sort_values(['crowdedness_percent'],ascending=True)
        res['bucket'] = pd.cut(res['crowdedness_percent'], 
                              bins=np.linspace(0, 100, 101),  # 创建100个等宽区间
                              labels=range(100),  # 桶标签0-99
                              include_lowest=True)  # 包含最小值
         
        # 按桶分组并计算y的均值
        bucket_means = res.groupby('bucket')['y'].mean().reset_index()
         
        # 绘制柱状图
        plt.figure(figsize=(20, 8))
         
        # 绘制柱状图
        bars = plt.bar(bucket_means['bucket'], bucket_means['y'], 
                      width=1,  # 每个桶宽度为1
                      edgecolor='black', 
                      linewidth=0.5)
        dense_ticks = np.arange(0, 101, 1)  # 70到100，每5个单位
        plt.xticks(dense_ticks, dense_ticks)
        plt.xticks(rotation=90)
        
    ####crowdedness_study2 动量，小-大市值，门限测试
    if args.task=='crowdedness_study2':   
        
        dpg=rqdatac.get_price(order_book_ids='000510.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        
        dpg.index = dpg.index.get_level_values(1)
        wpg.index = wpg.index.get_level_values(1)
        T=40
        dpg['dpg_moment40']=(dpg['close'] - dpg['close'].shift(T)) / dpg['close'].shift(T)
        wpg['wpg_moment40']=(wpg['close'] - wpg['close'].shift(T)) / wpg['close'].shift(T)
        
        res=wpg[['close','wpg_moment40']].join(dpg['dpg_moment40'])
        res['crowdedness']=res['wpg_moment40']-res['dpg_moment40']

        days=args.w*252
        res=res.iloc[-days:]
        

        from scipy.stats import percentileofscore
        res['crowdedness_percent'] = [percentileofscore(res['crowdedness'], v) for v in res['crowdedness']]
        
        res['y']=res['close'].shift(-1*args.t)/res['close']-1
        res=res.dropna()
        
        res=res.sort_values(['crowdedness_percent'],ascending=True)
        res['bucket'] = pd.cut(res['crowdedness_percent'], 
                              bins=np.linspace(0, 100, 101),  # 创建100个等宽区间
                              labels=range(100),  # 桶标签0-99
                              include_lowest=True)  # 包含最小值
         
        # 按桶分组并计算y的均值
        bucket_means = res.groupby('bucket')['y'].mean().reset_index()
         
        # 绘制柱状图
        plt.figure(figsize=(20, 8))
         
        # 绘制柱状图
        bars = plt.bar(bucket_means['bucket'], bucket_means['y'], 
                      width=1,  # 每个桶宽度为1
                      edgecolor='black', 
                      linewidth=0.5)
        dense_ticks = np.arange(0, 101, 1)  # 70到100，每5个单位
        plt.xticks(dense_ticks, dense_ticks)
        plt.xticks(rotation=90)
        
    ####crowdedness_study3 ，基差 门限测试
    if args.task=='crowdedness_study3':   
        
        # IM_basis=rqdatac.futures.get_dominant_price(underlying_symbols='IM',
        #                                     start_date=args.st,
        #                                     end_date=args.et,
        #                                     frequency='1d',fields=None,adjust_type='pre', adjust_method='prev_close_spread')
        
        # IM_basis=rqdatac.futures.get_basis(order_book_ids='IM2506', 
        #                                 start_date=args.st, 
        #                                 end_date=args.et,
        #                                 fields=None,frequency='1d')

        
        # IM_basis.index = IM_basis.index.get_level_values(1)
        # def func1(row):
        #     xx=rqdatac.futures.get_basis(order_book_ids=row.dominant_id, 
        #                             start_date=row.name, 
        #                             end_date=row.name,
        #                             fields=None,frequency='1d')
        #     return xx['basis_rate'].iloc[0]
        
        # IM_basis['basis']=IM_basis.apply(func1,axis=1)
        # IM_basis.to_csv('data/IM_basis.csv')
        
        IM_basis=pd.read_csv('data/IM_basis.csv',index_col=0,parse_dates=True)
        
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        wpg.index = wpg.index.get_level_values(1)
        
        days=args.w*252
        res=IM_basis.iloc[-days:]
        res['crowdedness']=res['basis'] 

        from scipy.stats import percentileofscore
        res['crowdedness_percent'] = [percentileofscore(res['crowdedness'], v) for v in res['crowdedness']]
        res=res[['crowdedness','crowdedness_percent']].join(wpg[['close']])
        
        res['y']=res['close'].shift(-1*args.t)/res['close']-1
        res=res.dropna()
        
        res=res.sort_values(['crowdedness_percent'],ascending=True)
        res['bucket'] = pd.cut(res['crowdedness_percent'], 
                              bins=np.linspace(0, 100, 101),  # 创建100个等宽区间
                              labels=range(100),  # 桶标签0-99
                              include_lowest=True)  # 包含最小值
         
        # 按桶分组并计算y的均值
        bucket_means = res.groupby('bucket')['y'].mean().reset_index()
         
        # 绘制柱状图
        plt.figure(figsize=(20, 8))
         
        # 绘制柱状图
        bars = plt.bar(bucket_means['bucket'], bucket_means['y'], 
                      width=1,  # 每个桶宽度为1
                      edgecolor='black', 
                      linewidth=0.5)
        dense_ticks = np.arange(0, 101, 1)  # 70到100，每5个单位
        plt.xticks(dense_ticks, dense_ticks)
        plt.xticks(rotation=90)
        
        x=a
        
  ####crowdedness_study4 换手率，小-大市值，门限测试
    if args.task=='crowdedness_study4':   
        
        dpg=rqdatac.get_price(order_book_ids='000510.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        
        dpg.index = dpg.index.get_level_values(1)
        wpg.index = wpg.index.get_level_values(1)
        
        res=pd.concat([wpg[['total_turnover','close']],dpg[['total_turnover','close']]],axis=1)
        res.columns=['wpg_turnover','wpg_close','dpg_turnover','dpg_close']
        res['turnover_ratio']=res['wpg_turnover']/res['dpg_turnover']
        
        from scipy.stats import percentileofscore
        def func1(window):
            return percentileofscore(window, window[-1])
        
        res['turnover_ratio_pct'] = res['turnover_ratio'].rolling(window=252*3, min_periods=252*3).apply(func1)
        
        res['wpg_y']=res['wpg_close'].shift(-1*args.t)/res['wpg_close']-1
        res['dpg_y']=res['dpg_close'].shift(-1*args.t)/res['dpg_close']-1
        res['y']=res['wpg_y']-res['dpg_y']
        res=res.dropna()
        
        res=res.sort_values(['turnover_ratio_pct'],ascending=True)
        res['bucket'] = pd.cut(res['turnover_ratio_pct'], 
                              bins=np.linspace(0, 100, 101),  # 创建100个等宽区间
                              labels=range(100),  # 桶标签0-99
                              include_lowest=True)  # 包含最小值
         
        # 按桶分组并计算y的均值
        bucket_means = res.groupby('bucket')['y'].mean().reset_index()
         
        # 绘制柱状图
        plt.figure(figsize=(20, 8))
         
        # 绘制柱状图
        bars = plt.bar(bucket_means['bucket'], bucket_means['y'], 
                      width=1,  # 每个桶宽度为1
                      edgecolor='black', 
                      linewidth=0.5)
        dense_ticks = np.arange(0, 101, 1)  # 70到100，每5个单位
        plt.xticks(dense_ticks, dense_ticks)
        plt.xticks(rotation=90)
        
  ####crowdedness_study5 估值，小-大市值，门限测试
    if args.task=='crowdedness_study5':   

        dpg_pe=rqdatac.index_indicator(['399300.XSHE'],
                                   start_date=args.st,end_date=args.et)
        wpg_pe=rqdatac.index_indicator(['399303.XSHE'],
                                   start_date=args.st,end_date=args.et)
        
        dpg_pe.index = dpg_pe.index.get_level_values(1)
        wpg_pe.index = wpg_pe.index.get_level_values(1)
        
        dpg=rqdatac.get_price(order_book_ids='000510.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        
        dpg.index = dpg.index.get_level_values(1)
        wpg.index = wpg.index.get_level_values(1)        
        
        
        res=pd.concat([wpg[['close']],dpg[['close']],wpg_pe[['pe_ttm']],dpg_pe[['pe_ttm']]],axis=1)
        res.columns=['wpg_close','dpg_close','wpg_pe','dpg_pe']
        res['pe_ratio']=res['wpg_pe']/res['dpg_pe']
        
        from scipy.stats import percentileofscore
        def func1(window):
            return percentileofscore(window, window[-1])
        
        res['pe_ratio_pct'] = res['pe_ratio'].rolling(window=252*3, min_periods=252*3).apply(func1)
        
        res['wpg_y']=res['wpg_close'].shift(-1*args.t)/res['wpg_close']-1
        res['dpg_y']=res['dpg_close'].shift(-1*args.t)/res['dpg_close']-1
        res['y']=res['wpg_y']-res['dpg_y']
        res=res.dropna()
        
        res=res.sort_values(['pe_ratio_pct'],ascending=True)
        res['bucket'] = pd.cut(res['pe_ratio_pct'], 
                              bins=np.linspace(0, 100, 101),  # 创建100个等宽区间
                              labels=range(100),  # 桶标签0-99
                              include_lowest=True)  # 包含最小值
         
        # 按桶分组并计算y的均值
        bucket_means = res.groupby('bucket')['y'].mean().reset_index()
         
        # 绘制柱状图
        plt.figure(figsize=(20, 8))
         
        # 绘制柱状图
        bars = plt.bar(bucket_means['bucket'], bucket_means['y'], 
                      width=1,  # 每个桶宽度为1
                      edgecolor='black', 
                      linewidth=0.5)
        dense_ticks = np.arange(0, 101, 1)  # 70到100，每5个单位
        plt.xticks(dense_ticks, dense_ticks)
        plt.xticks(rotation=90)
        
        x=a
        
    ####pick_st_sell 找出st的票然后清掉
    if args.task=='pick_st_sell':   
        df=pd.read_excel(args.ATX_pos_file,dtype=str)
        st=df[df['证券名称'].str.contains('ST')] ##筛选出st的
        st=st[st['持仓盈亏'].apply(float)>0] ##筛选出盈利的st卖掉
        
        st['证券市场']=st['交易市场'].apply(lambda x:'SZ' if x=='深交所' else 'SH')
        st['证券代码']=st['证券代码']+'.'+st['证券市场']
        st['当前拥股']=st['持仓数量'].apply(int)
        st['目标拥股']=st['持仓数量'].apply(lambda x: int(float(x) * args.ratio))
        st['调整股数']=st['目标拥股']-st['当前拥股']
        st['调整股数']=st['调整股数'].apply(lambda x: round(x / 100) * 100)
        st=st.sort_values('调整股数',ascending=True)
        st=st[st['调整股数']!=0]
        
        st['算法类型']='TWAP'
        # st['账户名称']='百榕全天候宏观对冲绝对收益信用'
        st['算法实例']='kf_twap_plus'
        st['证券代码']=st['证券代码']
        st['交易方向']=st['调整股数'].apply(lambda x:'买入' if x>0 else '卖出')
        st['任务数量']=st['调整股数'].apply(abs)
        st['开始时间']=args.st
        st['结束时间']=args.et
        st['涨跌停是否继续执行']='涨跌停继续交易'
        st['过期后是否继续执行']='否'
        st['其他参数']=np.nan
        st['交易市场']=np.nan
        
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
        st=st[columns]
        
        st.to_csv(args.ATX_file,index=False)        
        
        

        
    ####hthg_index 华泰宏观经济日频指标
    if args.task=='hthg_index':   
        df=pd.read_csv('wind_csv/华泰宏观经济日频指标.csv',skiprows=[1,2,3,4],index_col=['指标名称'],parse_dates=True)
        df=df['20160101':]
        df=df.fillna(method='ffill').fillna(method='bfill')    
        
        df['铜金比']=df['期货收盘价(连续):COMEX铜']/df['期货收盘价(连续):COMEX黄金']
        df['铜金比_ewm']=df['铜金比'].ewm(span=364).mean()
        
        df['波罗的海干散货指数']=df['波罗的海干散货指数(BDI)']
        df['波罗的海干散货指数_ma']=df['波罗的海干散货指数'].rolling(window=28).mean()
       
        df['PTA平均产业链负荷率']=(df['中国:开工率:精对苯二甲酸:PTA工厂']+df['中国:开工率:精对苯二甲酸:聚酯工厂']+df['中国:开工率:精对苯二甲酸:江浙织机'])/3
        df['PTA平均产业链负荷率_ma']=df['PTA平均产业链负荷率'].rolling(window=28).mean()
        
        df['建材综合指数']=df['中国:建材综合指数']
        df['建材综合指数_ma']=df['建材综合指数'].rolling(window=28).mean()
        
        df['秦皇岛港煤炭吞吐量']=df['秦皇岛港:煤炭调度:港口吞吐量']
        df['秦皇岛港煤炭吞吐量_ma']=df['秦皇岛港煤炭吞吐量'].rolling(window=28).mean()
        
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date='20150101', 
                  end_date='20250603', 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)     
        wpg.index = wpg.index.get_level_values(1)  
        wpg=wpg[['close']]
        df=df.join(wpg)
        df=df.fillna(method='ffill').fillna(method='bfill')    
        
        Plot.plot_res(df,'',cols = ['close','铜金比_ewm','波罗的海干散货指数_ma','PTA平均产业链负荷率_ma','建材综合指数_ma','秦皇岛港煤炭吞吐量_ma'],start_time = df.index[0],
                                        end_time=df.index[-1],
                                        days = None,
                                        maxmin=True)
        xx=a
        
    ####htldx_index 华泰流动性日频指标
    if args.task=='htldx_index':   
        df=pd.read_csv('wind_csv/华泰流动性维度指标.csv',skiprows=[1,2,3,4],index_col=['指标名称'],parse_dates=True)
        df=df['20160101':]
        df=df.fillna(method='ffill').fillna(method='bfill')    
        
        
        df['银行间质押式回购加权利率']=df['中国:银行间质押式回购加权利率:7天']
        df['银行间质押式回购加权利率_ewm']=df['银行间质押式回购加权利率'].ewm(span=364).mean()
        
        df['SHIBOR']=df['SHIBOR:3个月']
        df['SHIBOR_ewm']=df['SHIBOR'].ewm(span=364).mean()
        
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date='20150101', 
                  end_date='20250603', 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)     
        wpg.index = wpg.index.get_level_values(1)  
        wpg=wpg[['close']]
        df=df.join(wpg)
        df=df.fillna(method='ffill').fillna(method='bfill')    
        
        Plot.plot_res(df,'',cols = ['close','银行间质押式回购加权利率_ewm','SHIBOR_ewm'],start_time = df.index[0],
                                        end_time=df.index[-1],
                                        days = None,
                                        maxmin=True)
        xx=a
        
    ####wpg_compare wind,qmt,rq微盘股对比
    if args.task=='wpg_compare':   
        rq_wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date='20200101', 
                  end_date='20250613', 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)     
        rq_wpg.index = rq_wpg.index.get_level_values(1) 
        rq_wpg=rq_wpg['20220101':'20230101']
        # rq_wpg=rq_wpg['20220101':'20250613']
        rq_wpg['return']=rq_wpg['close']/rq_wpg['prev_close']-1
        # from xtquant import xtdata
        # # xtdata.get_stock_list_in_sector()
        # id='102722.BKZS'
        # # xtdata.download_history_data('102722.BKZS', period='1d', start_time='', end_time='')
        # qmt_wpg = xtdata.get_market_data_ex([],[id],period='1d',
        #                                  start_time = '20200101',count=-1,
        #                                  dividend_type='front_ratio')
        wind_wpg=pd.read_csv('wind_csv/8841431.WI.csv',index_col='日期',parse_dates=True)
        wind_wpg=wind_wpg['20220101':'20230101']
        # wind_wpg=wind_wpg['20220101':'20250613']
        
        print('米筐')
        Metrics.print_metrics(rq_wpg['return'],rq_wpg.index,0.03) 
        print('wind')
        Metrics.print_metrics(wind_wpg['涨跌幅'],wind_wpg.index,0.03) 
        x=1
        
        
    ####opt_wpg_hlg_bond 组合优化微盘股+红利+债券
    if args.task=='opt_wpg_hlg_bond':   
        st='20200101'
        et='20250626'

        wpg=rqdatac.get_price(order_book_ids='866006.RI',  ##微盘股
                  start_date=st, 
                  end_date=et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)             
        hlg=rqdatac.get_price(order_book_ids='000922.XSHG',  ##中证红利
                  start_date=st, 
                  end_date=et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)     
        bond=rqdatac.get_price(order_book_ids='932311.INDX',  ##中证7-30年国债及政策性金融债指数
                  start_date=st, 
                  end_date=et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)     
        wpg.index = wpg.index.get_level_values(1) 
        wpg['return']=wpg['close']/wpg['prev_close']-1
        hlg.index = hlg.index.get_level_values(1) 
        hlg['return']=hlg['close']/hlg['prev_close']-1
        bond.index = bond.index.get_level_values(1) 
        bond['return']=bond['close']/bond['prev_close']-1
        
        riskfolio_returns=pd.concat([wpg[['return']],hlg[['return']],bond[['return']]],axis=1)
        riskfolio_returns.columns=['wpg','hlg','bond']
        N=252*5
        riskfolio_returns=riskfolio_returns.iloc[-(N):]    
        
        asset_classes=pd.DataFrame(riskfolio_returns.columns,columns=['Assets'])
        asset_classes['Class 1']='stock'
        
        import riskfolio as rp
        def classic_mean_risk_optimization(returns,asset_classes,task,plot=True):
            '''
            Mean-Variance Portfolios和Mean-Risk Portfolios都是现代投资组合理论中的概念，用于帮助投资者在风险和回报之间找到最佳平衡。尽管它们在目标上相似，即在不同的风险水平上最大化回报，但它们在考虑风险的方式上存在差异。
    
            Mean-Variance Portfolios
            Mean-Variance Portfolios基于Harry Markowitz于1952年提出的现代投资组合理论，也称为均值-方差优化。这种方法的核心在于投资组合的选择不仅取决于其预期回报（均值）而且还取决于其风险（方差或标准差）。Markowitz的理论认为，通过分散投资组合，可以在不同的风险水平上最大化预期回报。在这种方法中，风险是通过投资组合回报的方差或标准差来衡量的，这反映了投资回报的波动性。
            
            Mean-Risk Portfolios
            Mean-Risk Portfolios也旨在平衡回报和风险，但它们在定义风险时可能采用不同于方差的其他风险度量。这些风险度量可以包括但不限于下行风险、Value at Risk (VaR)、Conditional Value at Risk (CVaR)、或其他风险度量标准。这种方法认识到不同的投资者可能对风险有不同的容忍度，特别是在对风险的不同方面更为敏感时（例如，更关心损失的可能性而不是收益的波动性）。
            '''
            if task=='estimating_mean_variance_portfolios':
                '''
                2.1 Calculating the portfolio that maximizes Sharpe ratio.
                '''
                # Building the portfolio object
                
                port = rp.Portfolio(returns=returns)  
                # Calculating optimal portfolio      
                # Select method and estimate input parameters:      
                method_mu='hist' # Method to estimate expected returns based on historical data.
                method_cov='hist' # Method to estimate covariance matrix based on historical data.        
                port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)   
                # Estimate optimal portfolio:   
                model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
                rm = 'MV' # Risk measure used, this time will be variance
                obj = 'MinRisk' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
                hist = True # Use historical scenarios for risk measures that depend on scenarios
                rf = 0 # Risk free rate
                l = 0 # Risk aversion factor, only useful when obj is 'Utility'     
                
                # constraints=pd.DataFrame([[False,'All Assets','','','<=',0.05,'','','',''],
                #                  [False,'All Assets','','','>=',0.01,'','','','']],
                #                 columns=['Disabled','Type','Set','Position','Sign','Weight','Type Relative','Relative Set','Relative','Factor'])
                # A, B = rp.assets_constraints(constraints, asset_classes)
                # port.ainequality = A
                # port.binequality = B
                w = port.optimization(model=model, rm=rm, kelly='approx',obj=obj, rf=rf, l=l, hist=hist)
                
                if plot:
                    '''
                    2.2 Plotting portfolio composition
                    '''
                    # Plotting the composition of the portfolio
                    ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                                     height=6, width=10, ax=None)
                    plt.show()            
                    '''
                    2.3 Calculate efficient frontier
                    '''        
                    points = 50 # Number of points of the frontier     
                    frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)  
                    # Plotting the efficient frontier       
                    label = 'Max Risk Adjusted Return Portfolio' # Title of point
                    mu = port.mu # Expected returns
                    cov = port.cov # Covariance matrix
                    returns = port.returns # Returns of the assets
                    ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                          rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                          marker='*', s=16, c='r', height=6, width=10, ax=None)                
                    plt.show()            
                    # Plotting efficient frontier composition
                    ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
                    plt.show()   
                return w  ##返回配置资产权重
            
        w=classic_mean_risk_optimization(riskfolio_returns, ##算出top票权重
                                                   asset_classes,
                                                   task='estimating_mean_variance_portfolios',
                                                   plot=True)      
        
        
        xx=1
    ####wpg_IM_hedge 微盘股和IM对冲
    if args.task=='wpg_IM_hedge':   
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None) 
        zz1000=rqdatac.get_price(order_book_ids='000852.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)   
        wpg.index = wpg.index.get_level_values(1)        
        zz1000.index = zz1000.index.get_level_values(1)    
        
        df=pd.concat([wpg['close'],zz1000['close']],axis=1)
        df.columns=['wpg','zz1000']
        df['wpg_return']=df['wpg']/df['wpg'].shift(1)-1
        df['zz1000_return']=df['zz1000']/df['zz1000'].shift(1)-1
        df['wpg_net']=df['wpg']/df['wpg'].iloc[0]
        df['zz1000_net']=df['zz1000']/df['zz1000'].iloc[0]
        df['net_return']=(df['wpg_return']-df['zz1000_return'])*0.6
        df['hedge_net']=list(Convert.returns_to_net(df['net_return']))
        df=df.dropna()
        
        Metrics.print_metrics(df['net_return'],df.index,0.03) 
        Plot.plot_res(df,'',cols = ['wpg_net','zz1000_net','hedge_net'],start_time = df.index[0],
                                        end_time=df.index[-1],
                                        days = None,
                                        maxmin=False)
        xx=1
        
    ####download 下载常用指数数据
    if args.task=='download':   
        #中证2000 '932000.INDX'
        #中证红利 '000922.XSHG'
        #米筐微盘股 '866006.RI'
        # A股指数 '000002.XSHG'
        # 中证港股通综合指数 '930930.INDX'
        # 国证机器人产业指数 '980022.INDX'
        # 中证A500 '000510.XSHG'
        # H50066.XSHG,"09:31-11:30,13:01-15:00",0.0,沪港AH溢价
        
        # args.id1='000001.XSHG' #上证
        # args.id2='399106.XSHE' #深证
        ids=['932000.INDX','000922.XSHG','866006.RI',
             '980022.INDX','000510.XSHG','000001.XSHG','399106.XSHE']
        
        for id in tqdm(ids):
            df=rqdatac.get_price(order_book_ids=id, 
                      start_date=args.st, 
                      end_date=args.et, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None) 
            df.index = df.index.get_level_values(1)    
            df.to_csv(f'data/{id}.csv')
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

    ###robot_ps 机器人指数ps估值
    if args.task=='robot_ps':   
        
        dates=rqdatac.get_trading_dates(args.st, args.et, market='cn')
        res=[] 
        import gc
        for date in tqdm(dates):           
            df=rqdatac.index_weights(order_book_id=args.id, date=date) ##中证红利
            df=df.reset_index()
            ps=rqdatac.get_factor(list(df['order_book_id']), 'ps_ratio_ttm', date,date)
            ps=ps.reset_index()
            ps2=round(ps['ps_ratio_ttm'].mean(),4)
            res.append([date,ps2])
            del df,ps,ps2
            gc.collect()
        res=pd.DataFrame(res,columns=['date','ps'])
        res=res.set_index('date')
        res['ps'].plot(rot=90)
        xx=a
    ####crowdedness2 机器人指数拥挤度
    if args.task=='crowdedness2':   
        
        #机器人指数拥挤度
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
            wpg=rqdatac.get_price(order_book_ids='980022.INDX', 
                      start_date=date, 
                      end_date=date, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            
            wpg_turnover=wpg['total_turnover'].iloc[0]
            total_turnover=r1['total_turnover'].iloc[0]+r2['total_turnover'].iloc[0]
            crowdedness=wpg_turnover/total_turnover
            res.append(crowdedness)
            
        wpg=rqdatac.get_price(order_book_ids='980022.INDX', 
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
        
    ####crowdedness3 小市值组的T日动量mS减去大市值组的T日动量mL，T取5/10/20/30/40/50/60
    if args.task=='crowdedness3':   
        
        dpg=rqdatac.get_price(order_book_ids='000510.XSHG', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                  start_date=args.st, 
                  end_date=args.et, 
                  frequency='1d', 
                  fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                  expect_df=True,time_slice=None)
        
        dpg.index = dpg.index.get_level_values(1)
        wpg.index = wpg.index.get_level_values(1)
        
        # Ts = [5, 10, 20, 30, 40, 50, 60]  
        Ts = [20]  
        def calculate_momentum(df, T):
            """计算T日动量（收益率）"""
            return (df['close'] - df['close'].shift(T)) / df['close'].shift(T)
        
        # 计算大盘股动量序列
        mL_series = {T: calculate_momentum(dpg, T) for T in Ts}
        # 计算微盘股动量序列
        mS_series = {T: calculate_momentum(wpg, T) for T in Ts}
        
        momentum_diff = {}
        for T in Ts:
            # 对齐时间索引（确保同一天数据相减）
            merged = pd.DataFrame({'mS': mS_series[T], 'mL': mL_series[T]}).dropna()
            momentum_diff[T] = merged['mS'] - merged['mL']
        
        res = pd.concat(
            [momentum_diff[T].rename(f'T={T}') for T in Ts],
            axis=1
        )
        
        res=res['20240101':]
        Plot.plot_res(res,'',cols = [f'T={T}' for T in Ts],start_time = res.index[0],
                                        end_time=res.index[-1],
                                        days = None,
                                        maxmin=False)
        xx=a

    ####cb_iv 计算可转债的隐含波动率
    if args.task=='cb_iv':   
        
        import QuantLib as ql

        def calculate_implied_volatility(
            option_type,  # 期权类型：'call' 或 'put'
            S,            # 标的资产价格(正股价格)
            K,            # 行权价
            T,            # 剩余期限（年）
            r,            # 无风险利率（年化）
            market_price, # 期权市场价格
            sigma_guess=1.0,  # 波动率初始猜测值（默认20%）
            max_iter=100,     # 最大迭代次数
            tol=1e-6          # 计算精度
        ):
            # 1. 设置定价引擎参数
            calendar = ql.China()  # 中国市场日历（根据标的资产所在市场调整）
            day_count = ql.Actual365Fixed()  # 日计数规则
            
            # 2. 构建期权对象
            option_type = ql.Option.Call if option_type == 'call' else ql.Option.Put
            payoff = ql.PlainVanillaPayoff(option_type, K)
            exercise = ql.EuropeanExercise(ql.Date().todaysDate() + ql.Period(int(T*365), ql.Days))
            option = ql.VanillaOption(payoff, exercise)
            
            # 3. 构建Black-Scholes模型参数
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
            r_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r, day_count))
            div_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, 0.0, day_count))  # 假设无股息
            vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, calendar, sigma_guess, day_count))
            
            # 4. 设置定价引擎
            bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_ts, r_ts, vol_ts)
            option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
            
            # 5. 计算隐含波动率（通过迭代调整vol_ts）
            iv = option.impliedVolatility(market_price, bsm_process, tol, max_iter)
            return iv
        
        # id="118057.XSHG"
        # id="123257.XSHE"
        # id="111023.XSHG"
        # id="123205.XSHE"
        id="113695.XSHG"
        cb_info=rqdatac.convertible.instruments(id)
        cb_info2=rqdatac.convertible.get_conversion_price(id)
        # cb_info2=rqdatac.convertible.get_call_info(id)

        cb_df=rqdatac.get_price(id, start_date=args.date, end_date=args.date,
                  frequency='1d', fields=None, adjust_type='pre',
                  skip_suspended =False, market='cn', expect_df=True,time_slice=None)
        stock_df=rqdatac.get_price(cb_info.stock_code, start_date=args.date, end_date=args.date,
                  frequency='1d', fields=None, adjust_type='pre',
                  skip_suspended =False, market='cn', expect_df=True,time_slice=None)
        
        option_type = 'call'    # 看涨期权
        S= stock_df['close'].iloc[0]# 标的价格（正股价格）
        K=cb_info2['conversion_price'].iloc[-1] #行权价(转股价)
        
        from datetime import datetime
        st=datetime.strptime(args.date, '%Y%m%d')
        et=cb_info.maturity_date
        delta_days = abs((et - st).days)
        T = delta_days / 365 # 剩余期限
        
        market_price = cb_df['close'].iloc[0] # 期权市场价格
        r=rqdatac.get_yield_curve(start_date=args.date, end_date=args.date)
        r=round(r['10Y'],4).iloc[0]
        
        
        def calculate_pure_bond_value(face_value=100, coupon_rate=0.002, discount_rate=0.0168, years_to_maturity=5.9, frequency=1):
            """
            计算纯债价值（现金流贴现模型）
            :param face_value: 债券面值（默认100元）
            :param coupon_rate: 票面利率（年化，默认0.2%）
            :param discount_rate: 折现率（年化，默认1.68%）
            :param years_to_maturity: 剩余期限（年，默认5.9）
            :param frequency: 付息频率（年付=1，半年付=2，默认1）
            :return: 纯债价值（保留2位小数）
            """
            total_pv = 0.0  # 总现值
            periods = int(years_to_maturity * frequency)  # 总付息期数
            remaining_years = years_to_maturity  # 剩余期限（年）
            
            # 1. 计算每期票息现金流现值
            coupon = face_value * coupon_rate / frequency  # 每期票息
            for t in range(1, periods + 1):
                # 第t期的折现因子 = 1 / (1 + r/frequency)^(t)
                discount_factor = 1 / (1 + discount_rate / frequency) ** t
                total_pv += coupon * discount_factor
            
            # 2. 计算本金偿还现值（最后一期）
            principal_discount_factor = 1 / (1 + discount_rate / frequency) ** periods
            total_pv += face_value * principal_discount_factor
            
            return round(total_pv, 2)

        # 代入参数计算
        pure_bond_value = calculate_pure_bond_value(
            face_value=100,
            coupon_rate=cb_info.coupon_rate,
            discount_rate=r, 
            years_to_maturity=T, 
            frequency=1  # 年付
        )
        
        
        
        
        
        
        # 计算隐含波动率
        iv = calculate_implied_volatility(option_type, S, K, T, r, (market_price-pure_bond_value)/(100/K))
        print(f"自算隐含波动率：{iv:.2%}")  
        
        indicators=rqdatac.convertible.get_indicators(id,start_date=args.date, end_date=args.date,fields=None)
        iv2=indicators['iv'].iloc[0]
        print(f"米筐隐含波动率：{iv2:.2%}")  
        
        xx=1
    ####wpg_liquidity 微盘股流动性
    if args.task=='wpg_liquidity':   
        # 存款准备金率
        reserve_ratio=rqdatac.econ.get_reserve_ratio(reserve_type='major',start_date='20150101',end_date='20250526')
        # 融资融券
        margin =rqdatac.get_securities_margin(['XSHE', 'XSHG'],
                                              start_date='20200101', end_date='20250526', 
                                              fields='margin_balance')
        margin.index = margin.index.get_level_values(1)
        # M1,M2..
        money=rqdatac.econ.get_money_supply(start_date='20200101', end_date='20250526')
        money['m1-m2-yoy']=money['m1_growth_yoy']-money['m2_growth_yoy']

    ####a_hk A股-港股涨幅，20天
    if args.task=='a_hk':   
        date='20250527'
        dates=rqdatac.get_trading_dates(start_date='20200101', end_date=date)
        res=[]
        
        for date in tqdm(dates):
            dates2=rqdatac.get_trading_dates(start_date='20190101', end_date=date)
            dates2=dates2[-20:]
            wpg=rqdatac.get_price(order_book_ids='000002.XSHG', 
                      start_date=dates2[0], 
                      end_date=date, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)  
            hlg=rqdatac.get_price(order_book_ids='930930.INDX', 
                      start_date=dates2[0], 
                      end_date=date, 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)  
            
            wpg_return30=wpg['close'].iloc[-1]/wpg['close'].iloc[0]-1
            hlg_return30=hlg['close'].iloc[-1]/hlg['close'].iloc[0]-1
            wpg_hlg_dif=wpg_return30-hlg_return30
            res.append(wpg_hlg_dif)
        
        res=pd.DataFrame(res,columns=['wpg_hlg_dif'])
        res = res.replace([-np.inf], np.nan).dropna()
        res=res[res['wpg_hlg_dif']<0.5]
        plt.plot(res['wpg_hlg_dif'])
        
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
        df=rqdatac.get_price(
                   # order_book_ids='932315.INDX', 
                    order_book_ids='866006.RI', 
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
            
            df=df['2025-04-07':]
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
            
            df['net']=df['net']/df['net'].iloc[0]
            df['MACD_net']=df['MACD_net']/df['MACD_net'].iloc[0]
            Plot.plot_res(df,'',cols = ["net",
                                        "MACD_net",
                                        ],start_time = df.index[0],
                                              end_time=df.index[-1],
                                              days = None,
                                              maxmin=False)
        ####macd+crowdness综合判断
        if args.method=='MACD_2+crowdness':
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
            
            r1=rqdatac.get_price(order_book_ids='000001.XSHG', 
                      start_date=df.index[0], 
                      end_date=df.index[-1], 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            r2=rqdatac.get_price(order_book_ids='399106.XSHE', 
                      start_date=df.index[0], 
                      end_date=df.index[-1], 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            wpg=rqdatac.get_price(order_book_ids='866006.RI', 
                      start_date=df.index[0], 
                      end_date=df.index[-1], 
                      frequency='1d', 
                      fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
                      expect_df=True,time_slice=None)
            r1.index = r1.index.get_level_values(1)
            r2.index = r2.index.get_level_values(1)
            wpg.index = wpg.index.get_level_values(1)
            
            df['crowdedness']= wpg['total_turnover']/(r1['total_turnover']+r2['total_turnover'])
            df['crowdedness_signal']=df['crowdedness']>0.02
            
            df['flag'] = False ##True代表该天空仓
            # 标记区间的开始和结束
            flag = False  
            for i in range(len(df)):
                index=df.index[i]
                if df.loc[index, 'MACD_signal']==-1 and df.loc[index, 'crowdedness_signal']==True and not flag:
                    flag = True
                # if df.loc[index, 'MACD_signal']==1 and flag:
                if df.loc[index, 'MACD_signal']==1 and df.loc[index, 'crowdedness_signal']==False and flag:
                    flag = False
                if flag:
                    df.loc[index, 'flag'] = True
            df['flag']=df['flag'].shift(1)
            df=df.dropna()
            
            def func2(row):
                if row['flag']:
                    return 0
                elif row.name.month in [1,3,4,12]:
                    return 0
                else:
                    return row['return']
            df['MACD_return']=df.apply(func2, axis=1)
            
            
            
            
            df=df['2025-04-07':]
            print(df.index[0],df.index[-1])
            def check_inconsistency(window):
                return window[0] != window[1]
            res = df['flag'].rolling(window=2).apply(check_inconsistency, raw=True)
            inconsistency_count = res.sum()
            print(f'交易次数：{inconsistency_count}')
            
            Metrics.print_metrics(df['return'],df.index,0.03)  
            df['net']=list(Convert.returns_to_net(df['return'])) 
            print('\n')
            Metrics.print_metrics(df['MACD_return'],df.index,0.03)   
            df['MACD_net']=list(Convert.returns_to_net(df['MACD_return'])) 
            
            df['net']=df['net']/df['net'].iloc[0]
            df['MACD_net']=df['MACD_net']/df['MACD_net'].iloc[0]
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

    ####read_h5 读h5文件
    if args.task=='read_h5':      
        import h5py
        with h5py.File(r'C:\Users\陈天宇\.rqalpha-plus\bundle\indexes.h5', 'r') as h5file:
            # 2. 查看文件中的数据集（可选，用于确认结构）
            print("H5文件中的数据集：", list(h5file.keys()))
            
            dataset = h5file['866006.RI']  # 例如 h5file['data']
            df = pd.DataFrame(dataset[:])
            xx=1
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    ####入参
    
    ####常用指数
    #中证2000 '932000.INDX'
    #中证红利 '000922.XSHG'
    #米筐微盘股 '866006.RI'
    # 中证港股通综合指数 '930930.INDX'
    # 国证机器人产业指数 '980022.INDX'
    # 930713.INDX,"09:31-11:30,13:01-15:00",0.0,中证人工智能主题指数
    # 399976.XSHE,"09:31-11:30,13:01-15:00",0.0,CS新能车
    # 中证A500 '000510.XSHG'
    # H50066.XSHG,"09:31-11:30,13:01-15:00",0.0,沪港AH溢价
    # 000985.XSHG,"09:31-11:30,13:01-15:00",0.0,中证全指
    # 399986.XSHE,"09:31-11:30,13:01-15:00",0.0,中证银行
    # 000905.XSHG,"09:31-11:30,13:01-15:00",0.0,中证500,XSHG
    # 000852.XSHG,"09:31-11:30,13:01-15:00",0.0,中证1000
    # 000300.XSHG,"09:31-11:30,13:01-15:00",0.0,沪深300
    # 399303.XSHE,"09:31-11:30,13:01-15:00",0.0,国证2000
    # 932000.INDX,"09:31-11:30,13:01-15:00",0.0,中证2000指数
    # 931080.INDX,"09:31-11:30,13:01-15:00",0.0,中证30年期国债指数
    
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
    
    # args.task='factor_study'
    # args.st='20100101'
    # args.et='20250707'    
    
    args.task='factor_study2'
    args.st='20180126' ##全周期，有数据的第一天
    args.et='20250731'    
    # args.universe='whole_market'
    # args.universe='000300.XSHG'   ##300
    # args.universe='000905.XSHG'   ##500
    # args.universe='000906.XSHG'   ##800
    # args.universe='000852.XSHG' ##1000
    args.universe='399303.XSHE' ##国证2000
    
    
    # args.st='20070205'   ##牛市1，有数据的第一天
    # args.et='20071016'    
    
    # args.st='20071016'   ##熊市1
    # args.et='20081030'    
    
    # args.st='20081030'   ##牛市2
    # args.et='20090804'    
    
    # args.st='20090804'   ##熊市2
    # args.et='20130624'    
    
    # args.st='20130624'   ##震荡市1
    # args.et='20140721'    
    
    # args.st='20140721'   ##牛市3
    # args.et='20150612'    
    
    # args.st='20150612'   ##熊市3
    # args.et='20160128'    
    
    # args.st='20160128'   ##牛市4
    # args.et='20180126'    
    
    # args.st='20180126'   ##熊市4
    # args.et='20190103'    
    
    # args.st='20190103'   ##牛市5
    # args.et='20210218'    
    
    # args.st='20210218'   ##熊市5
    # args.et='20240924'    
    
    # args.st='20240924'   ##牛市6
    # args.et='20250731'      ##至今
    
    # args.task='factor_study3'
    # args.st='20240924'
    # args.et='20250731'    
    # args.factor='size'
    
    # args.task='backtest1'
    # args.st='20170101'
    # args.et='20250905'    
    
    # args.task='factor_study4'
    # args.st='20170101'
    # # args.st='20250601'
    # args.et='20250905'    
    # args.window = 21*2
    # args.factor=['size','beta','earnings_quality','momentum','liquidity']
    
    # args.task='zzqz_study'
    # args.st='20200101'
    # args.et='20250707'
    
    # args.task='1000_2000_beta_study'
    # args.st='20250815'
    # args.et='20250919'
    
    # args.task='wpg_drop_study'
    # args.st='20200101'
    # args.et='20250616'
    
    # args.task='wpg_hlg'
    
    
    # args.task='wpg_compare'
    
    # args.task='wpg_maxdrop_study'
    # args.st='20170101'
    # args.et='20250611'
    
    # args.task='wpg_maxdrop_study2'
    # args.st='20170101'
    # args.et='20250611'
    
    # args.task='wpg_zz1000_beta'
    # args.st='20250619'
    # args.et='20250919'
    
    # args.task='yz_return_study'
    # args.st='20170101'
    # args.et='20250819'
    
    # args.task='wpg_hlg_return_study'
    # args.st='20170101'
    # args.et='20251012'
    
    
    
    # args.task='make_backtest_file1'
    # args.ids=[
    #           '000852.XSHG',
    #           '932000.INDX',
    #           # '866006.RI',
    #           ]
    
    # args.factors={
    #     'size':-0.4,
    #     'earnings_yield':0.2,
    #     'beta':0.2,
    #     'liquidity':-0.2,
    #     }
    # args.top_n=100
    # args.st='20230901'
    # args.et='20251016'
    # args.days=1   ##因子回看天数
    # args.f=5   ##调仓频率
    # args.file=r'data/增强等权周频.xlsx'
    
    
    # args.task='make_backtest_file3' ##自制因子
    # args.ids=[
    #           '000852.XSHG',
    #           '932000.INDX',
    #           # '866006.RI',
    #           ]
    
    # args.factors=[
    #     # 'size',
    #     'beta',
    #     # 'liquidity',
    #     ]
    # # args.st='20250819'
    # args.st='20230901'
    # args.et='20250919'
    # args.days=1   ##因子回看天数
    # args.f=5   ##调仓频率
    # args.file=r'data/增强等权周频.xlsx'
    
    
    # args.task='backtest'
    # args.st='20200101'
    # args.et='20250319'
    # args.file=r'data/米筐微盘股红利股择时等权日频_1412月空仓.xlsx'
    
    # args.task='buy_sell'
    # # args.method='MACD_2+crowdness'
    # args.method='MACD_2'
    # args.id1='000001.XSHG'
    # args.id2='399106.XSHE'
    # args.st='20250107'
    # args.et='20250828'
    
    # args.task='stratgy1'
    
    # args.task='hlg_dividend'
    # args.st='20150101'
    # args.et='20250715'
    
    
    # args.task='exp'
    # args.method='compare_rq_wind'
    # args.id1='000001.XSHG'
    # args.id2='399106.XSHE'
    # args.st='20210101'
    # args.et='20250410'
    
    # args.task='wpg_macd_pred'
    # args.st='20170101'
    # args.et='20250509'
    # args.file='signal/wpg_macd_pred.csv'
    
    # args.task='hthg_index'
    
    # args.task='htldx_index'
    
    # args.task='cb_iv'  ##计算可转债的隐含波动率
    # args.date='20250721'
    
    # args.task='rq_wpg_make_pms_csv'   ##size+liq
    # # args.id='000002.XSHG' ##全A
    # # args.id='000300.XSHG' ##沪深300
    # # args.id='399852.XSHE' ##中证1000
    # # args.id='000905.XSHG' ##中证500
    # args.id='866006.RI'
    # args.et=pd.to_datetime('20250815')
    # args.money=200e4
    # args.n=300 ##top多少票
    
    # args.task='rq_wpg_make_pms_csv2'     ##liq
    # # args.id='000002.XSHG' ##全A
    # # args.id='000300.XSHG' ##沪深300
    # # args.id='399852.XSHE' ##中证1000
    # # args.id='000905.XSHG' ##中证500
    # args.id='866006.RI'
    # args.et=pd.to_datetime('20250908')
    # args.money=230e4
    # args.n=300 ##top多少票
    
    # args.task='rq_wpg_make_pms_csv3'     ##liq
    # args.ids=[
    #           '000852.XSHG',
    #           '932000.INDX',
    #           # '866006.RI',
    #           ]
    
    # args.factors=[
    #     # 'size',
    #     'beta',
    #     'liquidity',
    #     ]
    # args.et=pd.to_datetime('20251009')
    # args.money=175e4
    
    
    
    
    
    # args.task='rq_wpg_adjust_ATX'
    # args.pms_file='PMS_csv/共同target_2025-10-09.xlsx' ##目标持仓
    # args.start_time='20251010T093000000'
    # args.end_time=  '20251010T093500000'  
    
    # args.ATX_pos_file='ATX_csv/持仓查询none.xlsx'  ##现有持仓
    # args.ATX_file='ATX_csv/ATX_stock_2025-10_10百里挑一信用.csv'
    # args.account='百榕百里挑一稳健一号信用'
    
    # args.ATX_pos_file='ATX_csv/全天候持仓查询_20250901090843.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-09-01_绝对收益信用.csv'
    # args.account='百榕全天候宏观对冲绝对收益信用'
    
    
    # args.task='ATX_to_PMS_track'
    # args.ATX_file='ATX_csv/成交查询绝对收益_20250429095802.xls'
    
    # args.task='ATX_to_ATX_adjust'
    # args.st='20251013T093000000'
    # args.et='20251013T100000000'  
    # args.ratio=0.8
    
    # args.ATX_pos_file='ATX_csv/持仓查询_20251013091427.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-10-13_百里挑一信用.csv'
    # args.account='百榕百里挑一稳健一号信用'
    
    # args.ATX_pos_file='ATX_csv/全天候持仓查询_20250904085849.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-09-04_绝对收益信用.csv'
    # args.account='百榕全天候宏观对冲绝对收益信用'
    
    # args.task='ATX_to_ATX_adjust2'   ##正盈利清仓
    # args.st='20250827T130000000'
    # args.et='20250827T133000000'  
    # args.ratio=0.95
    
    # args.ATX_pos_file='ATX_csv/百里挑一持仓查询_20250827121430.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-08-27_百里挑一信用.csv'
    # args.account='百榕百里挑一稳健一号信用'
    
    # args.ATX_pos_file='ATX_csv/持仓查询_20250731091312.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-07-31_绝对收益信用.csv'
    # args.account='百榕全天候宏观对冲绝对收益信用'
    
    # args.task='ATX_to_ATX_adjust3'   ##正盈利清仓
    # args.st='20250827T130000000'
    # args.et='20250827T133000000'  
    # args.ratio=0.95 ##保持多少仓位，0.95是降低5%仓位
    
    # # args.ATX_pos_file='ATX_csv/百里挑一持仓查询_20250827121430.xlsx'
    # # args.ATX_file='ATX_csv/ATX_stock_2025-08-27_百里挑一信用.csv'
    # # args.account='百榕百里挑一稳健一号信用'
    
    # args.ATX_pos_file='ATX_csv/全天候持仓查询_20250827131539.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-08-27_绝对收益信用.csv'
    # args.account='百榕全天候宏观对冲绝对收益信用'
    
    
    # args.task='opt_wpg_hlg_bond'
    
    # args.task='wpg_IM_hedge'
    # args.st='20190101'
    # args.et='20250701'
    
    # args.task='yz_study'
    # args.st='20150101'
    # args.et='20250818'
    
    # args.task='basis_study'
    # args.st='20200101'
    # args.et='20250905'
    
    # args.task='download'
    # args.st='20100101'
    # args.et='20250630'
    
    # args.task='compare_rq_cty_beta_factor'
    # args.et=20250915
    
    # args.task='compare_rq_cty_size_factor'
    # args.et=20250915
    
    # args.task='compare_rq_cty_liquidity_factor'
    # args.et=20250915
    
    # args.task='crowdedness'
    # args.id1='000001.XSHG'
    # args.id2='399106.XSHE'
    # args.st='20250401'
    # args.et='20250509'
    
    # args.task='robot_ps'
    # args.st='20180101'
    # args.et='20250901'
    # # args.id='980022.INDX' ##机器人
    # # args.id='930713.INDX' ##AI
    # args.id='399976.XSHE' ##新能源汽车
    # args.id='980017.XSHE' ##芯片
    
    # args.task='crowdedness2'
    # args.id1='000001.XSHG'
    # args.id2='399106.XSHE'
    # args.st='20250401'
    # args.et='20250701'
    
    # args.task='crowdedness3'
    # args.st='20230101'
    # args.et='20250609'
    
    # args.task='wpg_adjust_dif'
    # args.st='20240101'
    # args.et='20250414'
    
    # args.task='wpg_market_value_median'
    # args.et='20250410'
    
    # args.task='crowdedness_study1'
    # args.id1='000001.XSHG'
    # args.id2='399106.XSHE'
    # args.st='20200101'
    # args.et='20250609'
    # args.w=5
    # args.t=20
    
    # args.task='crowdedness_study4'
    # args.st='20160101'
    # args.et='20250708'
    # # args.w=5
    # args.t=90
    
    # args.task='crowdedness_study5'
    # args.st='20160101'
    # args.et='20250708'
    # # args.w=5
    # args.t=60
    
    # args.task='crowdedness_study3'
    # args.st='20200101'
    # args.et='20250609'
    # args.w=5
    # args.t=20
    
    # args.task='pick_st_sell'
    # args.ATX_pos_file='ATX_csv/持仓查询_20250508105218.xlsx'
    # args.ATX_file='ATX_csv/ATX_stock_2025-05-08.csv'
    # args.st='20250508T130000000'
    # args.et=  '20250508T131500000'  
    # args.ratio=0
    
    main(args)
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import logging
import warnings
warnings.filterwarnings('ignore')

from config.config import ConfigParser,stock_info,stock_info_path
from tools.convert_func import Convert  
from tools.metrics_func import Metrics
from tools.general_func import General
from tools.plot_func import Plot
from tools.riskfolio_func import Riskfolio

from xtquant import xtdata

class Analysis:
    def __init__(self):
        pass


    @staticmethod
    def valset_analysis(args,agent,valset):
        group=valset.groupby('id')
        res=[]
        for id,df in tqdm(group):
            df['e']=abs(df['pred']-df['gt'])
            df=df[(df['e']!=0) & (df['gt']!=0)]
            df['mape']=df['e']/abs(df['gt'])
            if str(df['mape'].mean())=='nan': 
                continue
            threshold = np.percentile(df['mape'], 90) ##去掉top10%最大值的干扰
            filtered_values = df[df['mape'] < threshold]['mape']
            res.append([id,filtered_values.mean()])
        res=sorted(res,key=lambda x:x[1])
        return res
            
      
    @staticmethod
    def neatural_backtest(args,agent,dataset): 
        ids=dataset['id'].unique() ##dataset中所包含id list
        id_num=len(ids) ##计算标的数量
        pre_pos=pd.DataFrame([0]*id_num,index=ids,columns=['pre_pos']) ##初始化pre_pos
        close_pos=pd.DataFrame([0]*id_num,index=ids,columns=['close_pos']) ##初始化close_pos
        
        dfs=[] ##保存包含position，position_change这些信息的新df
        returns=[] ##保存每个date的return
        fees=[] ##保存每个date的fee
        target_poss=[] 
        close_poss=[] 
        dates=[]
        caps=[] ##每天的策略容量
        count=0
        # pre_basis_flag=False ##基差调仓会用到
        
        dataset['date']=dataset.index
        group=dataset.groupby('date')
        for date,df in tqdm(group):
            
            if date<pd.to_datetime(args.start_time) or date>pd.to_datetime(args.end_time):  ##取选中的时间段
                continue
            
            count+=1 ##天数计数
            # if len(df)!=len(pre_pos):
            #     print(f"df与pre_pos维度不一致，检查数据！count={count}")
                
            
            flag1=True if count%args.adjust_period==1 else False ##调仓日
            flag2=True if count%args.rebalance_period==1 and count!=1 else False ##仓位再平衡日

            ##根据pred算出排位信号
            zz500_index=df[df['id']==args.bench_index_code]##获取zz500指数
            df=df[df['id']!=args.bench_index_code]##zz500指数不参与预测return排位
            df.loc[df['suspendFlag']==1,'pred']=np.nan ##停盘票不参与排位
            df['ranks'],_=Convert.num_to_ranks_index(df['pred']) ##按当天预测return排位
            df['signal']=Convert.top_ranks_to_long(df['ranks'],args.k) ##按排位给出信号    
            df=df.sort_values(['ranks'],ascending=True)
            df=General.add_stock_name(df) 
            df.to_csv('result/ranks.csv',index=False)
            df=pd.concat([df,zz500_index], ignore_index=True) ##排位后拼回zz500指数
            df.loc[df['id']==args.bench_index_code,'signal']=-1 ##zz500指数永远是做空信号
            df=df.set_index('id')
            
            ##记录每天的pre_pos
            df=df.join(pre_pos)
            
            ##根据不同策略算出每天的target_pos
            if flag1==False:
                df['target_pos']=pre_pos               

            elif flag1==True:
                columns=list(df[df['signal']==1].index) ##找出当日top票
                if args.pos_stratagy=='riskfolio': ##用riskfolio计算组合权重
                    riskfolio_returns=Riskfolio.cal_dynamic_returns(agent,columns,date,N=args.N2)
                    asset_classes=pd.DataFrame(columns,columns=['Assets'])
                    asset_classes['Class 1']='stock'
                    for col in riskfolio_returns.columns:
                        if (riskfolio_returns[col] == 0).all():# 检查当前列是否全为0
                            riskfolio_returns[col] = riskfolio_returns.apply(lambda row: np.mean(row), axis=1) # 用每一行的均值替换该列的值
                    w=Riskfolio.classic_mean_risk_optimization(riskfolio_returns, ##算出top票权重
                                                               asset_classes,
                                                               task='estimating_mean_variance_portfolios',
                                                               plot=False)
                    if w is not None: ## w有解
                        # w=w.reset_index() 
                        w['weights']=w['weights']*args.w1 ##因为riskfolio是按总权重1算的，但在这里股票总权重=args.w1
                        w.rename(columns={'weights': 'target_pos'}, inplace=True)   
                    else: ##有时候无解的情况下用等权
                        w=pd.DataFrame(columns,columns=['id'])
                        w['target_pos']=args.w1/args.k        
                            
                elif args.pos_stratagy=='equal_weight': ##组合等权重
                    w=pd.DataFrame(columns,columns=['id'])
                    if count==1: ##第一次建仓
                        w['target_pos']=args.w1/args.k    
                    elif count!=1: ##非第一次调仓
                        w['target_pos']=pre_pos.iloc[-1]/args.k ##因为指数pos随着涨跌已经变化，要完美对冲的话，需要调节股票部分pos  
                
                elif args.pos_stratagy=='equal_num': ##组合等数量
                    w=pd.DataFrame(columns,columns=['id'])
                    data = xtdata.get_market_data_ex([],columns,period='1d',count=-1)
                    prices=[data[id].iloc[-1]['close'] for id in columns]##查询该id不复权最新股价
                    total=sum(prices)
                    w['target_pos']= [p/total*args.w1 for p in prices]                  
                
                # df=pd.merge(df,w,on='id',how='left') ##将riskfolio算出的权重merge
                df=df.join(w)
                df.loc[df['signal']!=1,'target_pos']=0   ##非top票的target_pos=0
                if count==1: ##第一次建仓
                    df.loc[df.index==args.bench_index_code,'target_pos']=args.w2 ##zz500指数target_pos是初始args.w2
                elif count!=1: ##非第一次调仓
                    df.loc[df.index==args.bench_index_code,'target_pos']=pre_pos.iloc[-1] ##zz500指数target_pos是前一天的pre_pos

            ##基差控制
            # try:
            #     pre_trading_day=General.pre_trading_day(date)
            #     pre_day_basis='none'
            #     pre_day_basis=args.basis.loc[pre_trading_day,'basis'] ##前一日基差
            #     if pre_day_basis>=args.basis_hold_threshold:
            #         basis_flag=True 
            #     else:
            #         basis_flag=False
            # except:
            #     basis_flag=False
            #     print('pre_day_basis wrong:',pre_day_basis)
            
            # print(f'pre_basis_flag-basis_flag:{pre_basis_flag}-{basis_flag}')
            # if pre_basis_flag==True and basis_flag==True:
            #     basis_coe=1
            # elif pre_basis_flag==False and basis_flag==False:
            #     basis_coe=1
            # elif pre_basis_flag==False and basis_flag==True:
            #     basis_coe=0.9
            # elif pre_basis_flag==True and basis_flag==False:
            #     basis_coe=1/0.9
                
            # df.loc[df['id']==args.bench_index_code,'target_pos']*=basis_coe
            
            # pre_basis_flag=basis_flag
            
            ##是否定期再平衡,适用于中性
            # if flag2==True:
            #     columns=list(df[(df['target_pos']!=0) & (df.index!=args.bench_index_code)].index) ##找出当日top票
            #     if args.pos_stratagy=='riskfolio': ##用riskfolio计算组合权重
            #         riskfolio_returns=Riskfolio.cal_dynamic_returns(agent,columns,date,N=args.N2)
            #         asset_classes=pd.DataFrame(columns,columns=['Assets'])
            #         asset_classes['Class 1']='stock'
            #         w=Riskfolio.classic_mean_risk_optimization(riskfolio_returns, ##算出top票权重
            #                                                     asset_classes,
            #                                                     task='estimating_mean_variance_portfolios',
            #                                                     plot=False)   
            #         if args.w2==0.5: ##完美中性
            #             df['target_pos'].update(w['weights']*df.loc[args.bench_index_code,'target_pos']) ##用空头市值再平衡
            #         else: ##暴露beta
            #             long_pos=df['target_pos'].sum()-df.loc[args.bench_index_code,'target_pos']
            #             df['target_pos'].update(w['weights']*long_pos) ##用多头市值再平衡
                        
                    
            ## 计算当天收益、手续费
            df[['daily_returns','daily_fee']] = df.apply(lambda row: Metrics.calc_daily_return_zz500_neatural(args,
                                                                                            row.name, 
                                                                                            row['pre_pos'], 
                                                                                            row['target_pos'], 
                                                                                            row['close'], 
                                                                                            row['preclose'], 
                                                                                            row['open']),axis=1, result_type='expand')
            
            ## 计算收盘后的close_pos
            def func(row):
                if row.name==args.bench_index_code:
                    return row['target_pos']+(-row['daily_returns'])
                else:
                    if row['target_pos']!=0:
                        return row['target_pos']+row['daily_returns']
                    else:
                        return 0
            df['close_pos']=df.apply(lambda row: func(row),axis=1)
            
            ##记录每天关键信息
            portfolio_daily_return=df['daily_returns'].sum()
            returns.append(portfolio_daily_return)
            portfolio_daily_fee=df['daily_fee'].sum()
            fees.append(portfolio_daily_fee)
            cap=df[df['signal']==1]
            cap=sum(cap['amount'])*args.cap_ratio/1e8 ##按当天所有持仓成交额总和的cap_ratio比例算，以亿表示
            caps.append(cap)
            dfs.append(df)
            target_poss.append(df['target_pos'])   
            # close_poss.append(df['close_pos'])   
            dates.append(date)
            
            ##状态更新、传递
            close_pos=df[['close_pos']]
            pre_pos['pre_pos'].update(close_pos['close_pos']) ##前一天的close_pos需要传递给第二天当pre_pos
            
        target_poss=pd.concat(target_poss,axis=1) ##查看持仓变化
        target_poss.columns=dates
        # print('相对3%的无风险利率：')
        Metrics.print_metrics(returns,dates,args.rf)   

        return dfs,target_poss,pd.Series(returns),dates,fees,caps
    
    
    @staticmethod
    def model_study(model): ##查看树模型叶子结点信息
        # 获取模型的booster对象
        booster = model.booster_   ## booster.dump_model()
        res=[]
        tree_ids=[i for i in range(model._Booster.num_trees())]
        
        for tree_id in tree_ids:
            leaf_num = booster.params['num_leaves']
            for leaf_id in range(leaf_num):
                leaf_value=booster.get_leaf_output(tree_id,leaf_id)
                res.append([tree_id,leaf_id,leaf_value])
        res=pd.DataFrame(res,columns=['tree_id','leaf_id','leaf_value'])
        return res
                
    @staticmethod
    def stock_pool_filter(ids):        
        
        df=stock_info[stock_info['id'].isin(ids)] ##找到对应ids的stock info
        df = df.dropna(subset=['bk']) ##去掉无板块信息的stock
        filter_condition=['SW1轻工制造','SW1钢铁','SW1公用事业','SW1纺织服装','SW1休闲服务','SW1建筑材料','SW1建筑装饰','SW1采掘','SW1房地产','SW1交通运输']
        df=df[~df['bk'].isin(filter_condition)] ##不在黑五类板块中的stock
        ids=list(df['id'])
        
        ids_filtered=[]
        for id in tqdm(ids):
            config=ConfigParser(id) 
            df=pd.read_csv(config.fe_data_path,index_col=0,parse_dates=True)
            if df.index[0]<=pd.to_datetime('2015-1-1'): ##stock时长不够的过滤掉
                ids_filtered.append(id)
                
        return  ids_filtered##返回过滤后的stock ids
                
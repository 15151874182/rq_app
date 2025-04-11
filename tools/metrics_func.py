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
from scipy.stats import spearmanr

from tools.convert_func import Convert  


class Metrics:
    def __init__(self):
        pass


    @staticmethod
    def return_total(returns):
        '''
        总收益率
        Parameters
        ----------
        returns : list/array/dataframe，直接传入0.002这种，不要传0.2%
            每日或其它周期的收益率list.
            
        Returns
        -------
        float
            该周期内的收益率.
        '''
        ## R：用连续returns计算单期收益率，cumprod：连乘
        return_total = (pd.Series(returns)+1).cumprod().iloc[-1]-1 
        return return_total

    @staticmethod
    def return_annual(returns):
        '''
        年化收益率

        '''
        ## R：用连续returns计算单期收益率，cumprod：连乘
        return_total = Metrics.return_total(returns)
        T=len(returns) ##多少个交易日
        return_annual=Convert.total_to_annual(return_total,T) ##年化收益率            
        return return_annual

    @staticmethod
    def volatility_annual(returns):
        '''
        年化波动率
        Parameters
        ----------
        returns : list/array/dataframe，直接传入0.002这种，不要传0.2%
            每日或其它周期的收益率list.
            
        Returns
        -------
        float
            该周期内的波动率.
        '''
        return np.std(returns)*np.sqrt(252) ##年化波动率

    @staticmethod
    def sharpe_ratio_annual(returns, rf):
        '''
        年化夏普比率.
        Parameters
        ----------
        returns : list/array/dataframe，直接传入0.002这种，不要传0.2%
            每日或其它周期的收益率list.
        r : float
            无风险利率，0.03表示3%，一般用十年国债利率.

        Returns
        -------
        TYPE
            夏普比率.

        '''
        return_annual = Metrics.return_annual(returns)   ##年化收益率   
        volatility_annual=Metrics.volatility_annual(returns) ##年化波动率

        return (return_annual - rf) / volatility_annual

    @staticmethod
    def max_drawdown(returns):
        '''
        returns期间的最大回撤
        Parameters
        ----------
        returns : list/array/dataframe，直接传入0.002这种，不要传0.2%
            每日或其它周期的收益率list.
            
        Returns
        -------
        TYPE
            最大回撤百分比，正数.
        '''
        ## R：用连续returns计算单期收益率，cumprod：连乘
        net = (pd.Series(returns)+1).cumprod()
        ##给定数组 [3, 1, 7, 2, 5]，它的 cummax() 结果会是 [3, 3, 7, 7, 7]
        cumsum = net.cummax() 
        return max((cumsum-net)/cumsum)

    @staticmethod
    def drawdown_analysis(dates, returns):
        net = Convert.returns_to_net(returns)
        df = pd.DataFrame({'net': list(net)}, index=dates)
        df['cumsum']=df['net'].cummax() 
        df['drawdown']=(df['cumsum']-df['net'])/df['cumsum']
        drawdown=list(df['drawdown'])
        
        ranges = [] ##每次回撤的起始+终止时间
        in_sequence = False
        start_date = None
        for i in range(len(df)):
            if df.iloc[i]['drawdown'] != 0 and not in_sequence:
                # 发现新的 1 数列，记录起始日期
                start_date = df.index[i]
                in_sequence = True
            elif df.iloc[i]['drawdown'] == 0 and in_sequence:
                # 当前数列结束，记录终止日期并保存
                end_date = df.index[i - 1]
                ranges.append((start_date, end_date))
                in_sequence = False
        def func(x):
            return len(df[x[0]:x[1]])
        recover_times=list(map(func,ranges)) ##每次修复回撤的天数
        max_drawdown=max(df['drawdown']) ##最大回撤幅度
        max_drawdown_date=df[df['drawdown']==max_drawdown].index ##发生最大回撤对应date
        if len(max_drawdown_date) != 1:
            max_drawdown_date=max_drawdown_date[0]
        max_drawdown_recover_time='still not' ##如果还没有recover的提示
        for i,x in enumerate(ranges):
            if max_drawdown_date>=x[0] and max_drawdown_date<=x[1]:
                max_drawdown_recover_time=recover_times[i] ##最大回撤修复天数
                break
        
        return ranges, recover_times, max_drawdown, max_drawdown_date, max_drawdown_recover_time


    @staticmethod
    def calmar_ratio_annual(returns, rf):
        '''
        年化卡玛比率.
        Parameters
        ----------
        returns : list/array/dataframe，直接传入0.002这种，不要传0.2%
            每日或其它周期的收益率list.
        r : float
            无风险利率，0.03表示3%，一般用十年国债利率.

        Returns
        -------
        TYPE
            年化卡玛比率.

        '''
        return_annual = Metrics.return_annual(returns)   ##年化收益率   
        max_drawdown = Metrics.max_drawdown(returns)
        if max_drawdown == 0:
            return np.inf
        return (return_annual - rf) / max_drawdown

    @staticmethod
    def print_metrics(returns,dates,rf):
        ranges, recover_times, max_drawdown, max_drawdown_date, max_drawdown_recover_time=Metrics.drawdown_analysis(dates, returns)
        print(f'return_total:{Metrics.return_total(returns)}')
        print(f'return_annual:{Metrics.return_annual(returns)}')
        print(f'volatility_annual:{Metrics.volatility_annual(returns)}')
        print(f'sharpe_ratio_annual:{Metrics.sharpe_ratio_annual(returns,rf)}')
        print(f'max_drawdown:{max_drawdown}')
        print(f'max_drawdown_recover_time:{max_drawdown_recover_time}')
        print(f'calmar_ratio_annual:{Metrics.calmar_ratio_annual(returns,rf)}')

    @staticmethod
    def output_metrics(id,returns,rf,start_time,end_time):
        output=[id,
                start_time,
                end_time,
                Metrics.return_total(returns),
                Metrics.return_annual(returns),
                Metrics.volatility_annual(returns),
                Metrics.sharpe_ratio_annual(returns,rf),
                Metrics.max_drawdown(returns),
                Metrics.calmar_ratio_annual(returns,rf)]
        output=pd.DataFrame(output).transpose()
        output.columns=['id','start_time','end_time','return_total',
                        'return_annual','volatility_annual',
                        'sharpe_ratio_annual','max_drawdown','calmar_ratio_annual']
        return output
        
    
    @staticmethod
    def acc_mape(res):
        error=abs(res['pred']-res['gt'])
        return 1-(error/res['gt']).mean()

    @staticmethod
    def mae(res):
        error=abs(res['pred']-res['gt']) 
        error=error.dropna()
        error=error[error!=0] ##去掉-99的预测
        return error.mean()
    
    @staticmethod
    def mse(res,col1='pred',col2='gt'):
        res=np.mean((res[col1]-res[col2]) **2)
        return res
    
    @staticmethod
    def corr(res,col1='pred',col2='gt'):
        res=res[col1].corr(res[col2])
        return res

    @staticmethod
    def acc(res,col1='pred',col2='gt'):
        same_sign = (res[col1] >= 0) == (res[col2] >= 0)
        return same_sign.mean() 
        
    @staticmethod
    def rank_acc(res,col1='pred',col2='gt',top_k=None):
        res['rank_pred'],_=Convert.num_to_ranks_index(res['pred'])
        res['rank_gt'],_=Convert.num_to_ranks_index(res['gt'])
        if top_k:
            res = res.nsmallest(top_k, 'rank_gt')
        spearman_corr, _ = spearmanr(res['rank_gt'], res['rank_pred'])
        return spearman_corr

    # @staticmethod
    # def calc_daily_return_zz500_neatural(args,
    #                                       w1, ##个股仓位权重
    #                                       w2, ##股指期货仓位权重，要负值
    #                                       pre_position,
    #                                       position,
    #                                       close,
    #                                       preclose,
    #                                       open):
    #     if pre_position==1:
    #         if position==1:##继续持仓
    #             daily_return=((close/preclose-1)-0)*w1
    #             daily_fee=0
    #         elif position==0:##开盘价卖掉
    #             daily_return=((open/preclose-1)-args.stock_sell_fee)*w1
    #             daily_fee=args.stock_sell_fee*w1
    #         elif position==-1: ##hs300中性中没有这种情况
    #             pass

    #     if pre_position==0:
    #         if position==1:##开盘价买
    #             daily_return=((close/open-1)-args.stock_buy_fee)*w1
    #             daily_fee=args.stock_buy_fee*w1
    #         elif position==0:##继续空仓
    #             daily_return=0
    #             daily_fee=0
    #         elif position==-1: ##开盘买入hs300股指期货空单
    #             daily_return=(-1*(close/open-1)-args.trade_fee_ratio)*w2
    #             daily_fee=args.trade_fee_ratio*w2

    #     if pre_position==-1: 
    #         if position==1: ##hs300中性中没有这种情况
    #             pass
    #         elif position==0: ##开盘卖出hs300股指期货空单
    #             daily_return=(-1*(open/preclose-1)-args.trade_fee_ratio)*w2
    #             daily_fee=args.trade_fee_ratio*w2
    #         elif position==-1:
    #             daily_return=(-1*(close/preclose-1)-0)*w2
    #             daily_fee=0
                
    #     return daily_return,daily_fee


    @staticmethod
    def calc_daily_return_zz500_neatural(args,
                                          id,
                                          pre_pos,
                                          target_pos,
                                          close,
                                          preclose,
                                          open):
        
        r1=(open/preclose-1)*pre_pos ##pre_pos部分能吃到开盘价收益
        r2=(close/open-1)*target_pos ##target_pos部分能吃到收盘价收益
        daily_fee=np.nan
        if id==args.bench_index_code: ##股指期货part
            if target_pos>=pre_pos: ##目标大于现有仓位，买入
                daily_fee=args.trade_fee_ratio*(target_pos-pre_pos) ##手续费是pos差额部分
            elif target_pos<pre_pos: ##目标小于现有仓位，卖出
                daily_fee=args.trade_fee_ratio*(pre_pos-target_pos) ##手续费是pos差额部分
            daily_return=-(r1+r2)-daily_fee ##期货做空，所以r去负
            
        elif id!=args.bench_index_code: ##股票part
            if target_pos>=pre_pos: ##目标大于现有仓位，买入
                daily_fee=args.stock_buy_fee*(target_pos-pre_pos) ##手续费是pos差额部分
            elif target_pos<pre_pos: ##目标小于现有仓位，卖出
                daily_fee=args.stock_sell_fee*(pre_pos-target_pos) ##手续费是pos差额部分
            daily_return=r1+r2-daily_fee ##当日总收益考虑这3个部分收益
                
        return daily_return,daily_fee
    
# Example usage:n
if __name__ == "__main__":
    # returns=[0.1,0.2,-0.15,-0.3,0.4,0.1] ##6天的return list
    returns=[0.1,0.2,0.15,0.3,0.4,-0.1] ##6天的return list
    net=[1,1.1,1.2,1.3,1.4,1.5] ##6天的净值
    rf = 0.02  # 无风险利率
    xx = Metrics()
    

    print("R:", xx.ret(returns))
    print("Volatility:", xx.volatility(returns))
    print("Sharpe Ratio:", xx.sharpe_ratio(returns, rf))
    print("Max Drawdown:", xx.max_drawdown(returns))
    print("Calmar Ratio:", xx.calmar_ratio(returns, rf))

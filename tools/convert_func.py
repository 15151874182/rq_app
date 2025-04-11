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
import dbf
import concurrent.futures
from datetime import datetime

from pandas_market_calendars import get_calendar
# from config.config import ConfigParser

def task1(id,res,agent): ##给多进程用的，task函数不能写在函数内部
    config=ConfigParser(id)
    raw_data=agent.read_fe_data(config)
    raw_data=raw_data[res.index[0].strftime('%Y-%m-%d'):res.index[-1].strftime('%Y-%m-%d')]
    raw_data=raw_data[['open','close','preclose','amount']]
    df=res.join(raw_data)
    df['id']=id
    return df      

class Convert:
    def __init__(self):
        pass

    @staticmethod
    def returns_to_net(returns):
        '''
        连续收益转连续净值
        Parameters
        ----------
        returns : list/array/dataframe，直接传入0.002这种，不要传0.2%
            每日或其它周期的收益率list.
            
        Returns
        -------
        pd.Series
            净值list.
        '''
        ##cumprod：单期return+1后，连乘
        return (pd.Series(returns)+1).cumprod()

    @staticmethod
    def net_to_returns(net):
        '''
        连续净值转连续收益
        Parameters
        ----------
        net : list/array/dataframe，net=[1,1.1,1.2,1.3,1.4,1.5] ##6天的净值
            净值list.
        Returns
        -------
        pd.Series
            收益率list.

        '''
        net=pd.Series(net)/net[0]
        returns=[]
        for idx,k in enumerate(net):
            if idx==0:
                returns.append(0) ##净值第一天都是1，所以return=0
            else:
                returns.append(net[idx]/net[idx-1]-1)
        return pd.Series(returns)
    
    @staticmethod
    def date_index_shift(data):
        ## data有时间索引
        time_index=data.index
        next_trading_day=Convert.next_trading_day(data.index[-1].date())
        time_index=time_index.append(pd.DatetimeIndex([next_trading_day]))
        time_index=time_index[1:]
        data.index=time_index       
        return data

    @staticmethod
    def num_to_ranks_index(nums):
        ##按num的大小，从高到低返回对应索引,[0.1,0.3,-0.1] 返回[1,0,2],因为0.3最大，索引是1
        nums = pd.Series(nums)
        ranks = list(nums.rank(ascending=False)) ##输入nums的对应排位
        index=[i for i in range(len(ranks))]
        item=list(zip(ranks,index)) ##组合ranks和index
        item=sorted(item,key=lambda x:x[0]) ##ranks数值越小，排位越大，第一>第二
        _, ranks_index=zip(*item)
        return list(ranks),list(ranks_index)

    @staticmethod
    def total_to_annual(return_total,T):      
        ##T的交易日总收益为return_total，转成年化收益率return_annual
        return (1+return_total)**(252/T)-1

    @staticmethod
    def ranks_to_long_short(ranks):
        ##输入一个多标的ranks排位，按中位数划分多空分组，1为做多，-1为做空
        long_short_list=[-1 if i>max(ranks)/2 else 1 for i in ranks]
        return long_short_list

    @staticmethod
    def top_ranks_to_long(ranks,k):
        ##输入一个多标的ranks排位，topk的排位做多=1，其它不持有=0
        long_list=[1 if i<=k else 0 for i in ranks]
        # long_list=[1 if (i>50 and i<=100) else 0 for i in ranks] ##分层回测的第二层，与top层收益差距就很大了，可以验证策略的有效性
        # long_list=[1 if (i>max(ranks)-50 and i<=max(ranks)) else 0 for i in ranks] ##分层回测的倒数层，可以用于做空
        return long_list
    
    @staticmethod
    def ress_to_dataset(agent,ress,save_path,val_test_flag=True):
        dataset=[]
  
        for id,df in ress:
            df['id']=id
            dataset.append(df)  
        dataset=pd.concat(dataset,axis=0)
        dataset.to_csv(save_path)
        return dataset


    @staticmethod
    def state_to_next(position,signal,change_flag):
        ##这个是持仓状态的转移方程，非常重要！！！！！！
        if change_flag:
            return signal
        else:
            return position

    @staticmethod
    def  time_to_full(data,start_time,end_time,fill_value=-99):##空缺值填-99，因为如果pred列是-99的话，就不会出现在top排位
        # 创建一个包含起始时间和终止时间范围的日期索引
        time_index = pd.date_range(start=start_time, end=end_time, freq='D')
        # 使用reindex函数补齐缺失的日期索引
        data = data.reindex(time_index,fill_value=fill_value)        

    @staticmethod
    def  timestamp_to_date(timestamp): 
        # 13位时间戳转普通时间，1710226800000 这种，10位的话不用/1000
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp/1000))

    @staticmethod
    def  wind_pms_to_nowday_pos(wind_pms): ##万德交易清单转持仓信息
        wind_pms=wind_pms[wind_pms['买卖方向'].isin(['买入'])]
        wind_pms.rename(columns={'证券代码': 'id','买卖数量': 'volume'}, inplace=True)
        wind_pms=wind_pms[['id','volume']]
        wind_pms['can_use_volume']=wind_pms['volume']
        nowday_pos=wind_pms
        nowday_pos=nowday_pos.reset_index(drop=True)
        return nowday_pos

    @staticmethod
    def df_to_dbf(df,dbf_path):
        fields = "; ".join([f"{col} C(64)" for col in df.columns])
        # 创建 DBF 表
        table = dbf.Table(dbf_path, fields)
        table.open(dbf.READ_WRITE)
        # 将 DataFrame 中的每一行写入 DBF 文件
        for row in df.itertuples(index=False, name=None):
            table.append(row)
        # 关闭 DBF 表
        table.close()   


    @staticmethod
    def date_to_timestamp(date):
        beijing_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        # 转换为时间戳
        timestamp = int(time.mktime(beijing_time.timetuple()))
        return timestamp
    
# Example usage:
if __name__ == "__main__":
    # returns=[0.1,0.2,-0.15,-0.3,0.4,0.1] ##6天的return list
    returns=[0.1,0.2,0.15,0.3,0.4,-0.1] ##6天的return list
    net=[1,1.1,1.2,1.3,1.4,1.5] ##6天的净值
    xx = Convert()
    
    print("net:", xx.returns_to_net(returns))
    print("returns:", xx.net_to_returns(net))


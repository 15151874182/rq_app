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
import math
from datetime import datetime,timedelta
import re

from xtquant import xtdata

from pandas_market_calendars import get_calendar
# from config.config import ConfigParser,stock_info_path,stock_info


class General:
    def __init__(self):
        pass

    @staticmethod
    def neighbor_trading_day(date=None):
        '''
        eg: date='2024-9-30'
        使用迅投api判断当前日是不是交易日，获取上一个，下一个交易日，比外面的api靠谱
        '''
        ## 
        if date:
            current_date = datetime.strptime(date, '%Y-%m-%d').date()
        else:
            current_date = datetime.now().date()
        start_date=current_date - timedelta(days=30)
        end_date=current_date + timedelta(days=30)
        
        
        current_date = current_date.strftime('%Y%m%d')
        start_date = start_date.strftime('%Y%m%d')
        end_date = end_date.strftime('%Y%m%d')
        
        trading_days=xtdata.get_trading_calendar(market='SH', 
                                                 start_time = start_date, 
                                                 end_time = end_date)
        try:
            id=trading_days.index(current_date)
        except:
            is_trading_day=False
            pre_trading_day=None
            next_trading_day=None
            return is_trading_day, pre_trading_day, next_trading_day
        
        is_trading_day=True
        pre_trading_day=trading_days[id-1]
        pre_trading_day = f"{pre_trading_day[:4]}-{pre_trading_day[4:6]}-{pre_trading_day[6:]}"
        next_trading_day=trading_days[id+1]
        next_trading_day = f"{next_trading_day[:4]}-{next_trading_day[4:6]}-{next_trading_day[6:]}"
        
        return is_trading_day, pre_trading_day, next_trading_day

    @staticmethod
    def add_noise_resample(x, y, x_noise_level=0.01, y_noise_level=0.01, n_samples=2): ##扩充训练集，add noise
        # x: 特征数据，pandas DataFrame 格式
        # y: 标签数据，pandas Series 格式
        # x_noise_level: 添加的噪声比例
        # n_samples: 为每个样本生成的新样本数量
        
        x_resampled = [x]  # 保留原始数据
        y_resampled = [y]
        
        # 为每一行生成 n_samples 个带噪声的新样本
        for _ in range(n_samples):
            # 对特征添加噪声
            noise_X = x_noise_level * np.random.normal(size=x.shape)
            x_noisy = x + noise_X  # 添加噪声到特征
            x_resampled.append(pd.DataFrame(x_noisy, columns=x.columns))  # 将生成的新样本添加到列表中
            
            # 对标签添加噪声
            noise_y = y_noise_level * np.random.normal(size=y.shape)
            y_noisy = y + noise_y  # 添加噪声到标签
            y_resampled.append(pd.Series(y_noisy, index=y.index))  # 保留索引一致性
    
        # 将所有DataFrame和Series拼接成一个新的DataFrame和Series
        x_resampled = pd.concat(x_resampled, ignore_index=True)
        y_resampled = pd.concat(y_resampled, ignore_index=True)
        
        return x_resampled, y_resampled    

    @staticmethod
    def get_res(pred,gt):
        date_index=gt.index
        pred=np.array(pred).reshape(-1)
        gt=np.array(gt).reshape(-1)
        res=pd.DataFrame(index=date_index)
        res['pred'],res['gt']=pred,gt
        return res    

    @staticmethod
    def move_down_date(df): ##在预测中，输入x的date索引是当日的，输出y的date索引需要变成第二日的
        
        # next_trading_day=pd.to_datetime(General.next_trading_day(date=df.index[-1])) ##索引和特征是同一天，但pred和gt是代表第二天开始的结果，这个很重要！！
        _, _, next_trading_day=General.neighbor_trading_day(date=df.index[-1].strftime('%Y-%m-%d'))
        dates=list(df.index)[1:]
        dates.append(next_trading_day)
        df.index=dates           
        return df

    @staticmethod
    def factorial(n,k): ##组合的数量
        return math.factorial(n)/math.factorial(k)/math.factorial(n-k)

    @staticmethod
    def find_nan_pos(df): ##查找df存在nan的位置
        nan_positions = df.isna().stack()
        nan_positions = nan_positions[nan_positions]
        return nan_positions.index
    
    @staticmethod
    def add_stock_name(df,id='id'): ##df只有id，没有name列时候，加上
        xx=stock_info[['id','name']]
        res=pd.merge(df,xx,on=id,how='inner')
        return res
        
    @staticmethod
    def add_stock_name(df,id='id'): ##df只有id，没有name列时候，加上
        xx=stock_info[['id','name']]
        res=pd.merge(df,xx,on=id,how='inner')
        return res

    @staticmethod
    def normalize_list(input_list, lower_bound=0, upper_bound=100):
        """
        对输入的列表进行归一化处理，使其值位于指定的上下限范围内。
    
        :param input_list: 输入的列表，包含需要归一化的数值
        :param lower_bound: 归一化后的下限值，默认为 0
        :param upper_bound: 归一化后的上限值，默认为 100
        :return: 归一化后的列表
        """
        # 找出列表中的最小值和最大值
        min_val = min(input_list)
        max_val = max(input_list)
    
        # 若最大值和最小值相等，说明列表中所有元素相同，直接返回全为下限值的列表
        if max_val == min_val:
            return [lower_bound] * len(input_list)
    
        # 进行归一化操作
        normalized_list = [((i - min_val) / (max_val - min_val)) * (upper_bound - lower_bound) + lower_bound for i in input_list]
        return normalized_list
    
    @staticmethod
    def softmax_normalize(input_list):
        # 计算每个元素的指数值
        exp_values = [math.exp(i) for i in input_list]
        # 计算指数值的总和
        exp_sum = sum(exp_values)
        # 计算归一化后的结果
        normalized_list = [i / exp_sum for i in exp_values]
        return normalized_list

    @staticmethod
    def sum_normalize(input_list):
        normalized_list = [i/sum(input_list) for i in input_list]
        return normalized_list
    
    @staticmethod    
    def split_dataframe_by_index(df):
        # 定义匹配英文和中文的正则表达式
        english_pattern = re.compile(r'^[a-zA-Z_]+$')
        chinese_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')
     
        # 筛选出索引为英文的行
        english_df = df[df.index.map(lambda x: bool(english_pattern.match(x)))]
        # 筛选出索引为中文的行
        chinese_df = df[df.index.map(lambda x: bool(chinese_pattern.match(x)))]
     
        return english_df, chinese_df    
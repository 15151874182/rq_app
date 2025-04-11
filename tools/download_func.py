import os,sys,time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import traceback
from copy import deepcopy 

from config.config import ConfigParser,stock_info
import baostock as bs
from xtquant import xtdata
# 获取app路径=============================================================================
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 加入app路径系统环境变量
sys.path.insert(0,project_dir)

class Download:
    def __init__(self):
        bs.login()

    @staticmethod
    def get_stock_list(name):
        if name=='sz50': ##上证50
            rs=bs.query_sz50_stocks()
        elif name=='hs300': ##沪深300
            rs=bs.query_hs300_stocks()
        elif name=='zz500': ##中证500
            rs=bs.query_zz500_stocks()   
        elif name=='all': ##全部股票
            rs=bs.query_all_stock()
        else:
            raise KeyError(f'{name} not found')
            
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        ids = pd.DataFrame(data_list, columns=rs.fields)
        ids=list(ids['code'] + '_' + ids['code_name'])  
        return ids
    
    @staticmethod
    def download_data(config):
        
        # if type(stock_info) == str:
        #     if stock_info=='all': ##全部股票
        #         rs=bs.query_all_stock()
        #     elif stock_info=='sz50': ##上证50
        #         rs=bs.query_sz50_stocks()
        #     elif stock_info=='hs300': ##沪深300
        #         rs=bs.query_hs300_stocks()
        #     elif stock_info=='zz500': ##中证500
        #         rs=bs.query_zz500_stocks()
                
        #     stocks = []
        #     while (rs.error_code == '0') & rs.next():
        #         stocks.append(rs.get_row_data()[1:])
        # else:
        #     stocks=stock_info
            
        # # 循环下载stocks中每个成分股数据并保存到CSV文件
        # for stock in tqdm(stocks):
        stock_code, stock_name = config.id.split('_')# 股票代码和名称

        # 获取股票历史数据
        rs = bs.query_history_k_data_plus(code=stock_code, 
                                          fields='date,open,high,low,close,preclose,volume,amount,pctChg',
                                          start_date='2005-01-01', 
                                          end_date='2050-12-31', 
                                          frequency=config.frequency, ##数据频率如'd'表示按天.
                                          adjustflag=config.adjustflag) ##"2"后复权，“3”前复权.
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        # 生成DataFrame并保存为CSV文件
        df = pd.DataFrame(data_list, columns=rs.fields)
        raw_data_path = os.path.join(project_dir,config.raw_data_path)
        df.to_csv(raw_data_path, index=False)                





        
# Example usage:
if __name__ == "__main__":
    
    downloader = Download() ##需要这步去登录
    
    ##自定义成分股
    # ids=['sh.000016_上证50指数',
    #  'sh.000300_沪深300指数',
    #  'sh.000852_中证1000指数',
    #  'sh.000905_中证500指数']
    # ids=['sh.932000_中证2000指数']
    # 获取全市场成分股列表
    # ids=downloader.get_stock_list(name='all') ##有时候会失败
    # ids=list(stock_info['id'])
    # 获取上证500成分股列表
    # ids=downloader.get_stock_list(name='sz50')
    
    # 获取沪深300成分股列表
    ids=downloader.get_stock_list(name='hs300')
    
    # 获取中证500成分股列表
    # ids=downloader.get_stock_list(name='zz500')


    for id in tqdm(ids):
        try:
            config=ConfigParser(id) 
            Download.download_data(config)  
        except:
            traceback.print_exc()  
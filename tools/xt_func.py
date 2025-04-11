import os
import sys
import time
import joblib,re
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import traceback
import logging
import warnings
warnings.filterwarnings('ignore')

from config.config import ConfigParser,stock_info_path,stock_info

from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from xtquant import xtdata

class XT:
    def __init__(self):
        pass

    @staticmethod
    def download_by_xtquant(ids=['601288.SH','000001.SZ']):
        # 设定获取数据的周期
        period = "1d"
        # 下载标的行情数据
        ## 为了方便用户进行数据管理，xtquant的大部分历史数据都是以压缩形式存储在本地的
        ## 比如行情数据，需要通过download_history_data下载，财务数据需要通过
        ## 所以在取历史数据之前，我们需要调用数据下载接口，将数据下载到本地
        for id in ids:
            xtdata.download_history_data(id,period=period,incrementally=True) # 增量下载行情数据（开高低收,等等）到本地
        
        # xtdata.download_financial_data(ids) # 下载财务数据到本地
        # xtdata.download_sector_data() # 下载板块数据到本地
        # 更多数据的下载方式可以通过数据字典查询
        # 读取本地历史行情数据

    def sector_func():
        xx=xtdata.get_sector_list()
        bks=xtdata.get_stock_list_in_sector('迅投一级行业板块指数') #二级、三级
        for bk in bks:
            print(xtdata.get_instrument_detail(bk)['InstrumentName'])
        
    @staticmethod
    def getdata_by_xtquant(ids=['601288.SH','000001.SZ'],period = "1d"):
        # 下载标的行情数据
        history_data = xtdata.get_market_data_ex([],ids,period=period,count=-1)
        return history_data
    
    ####获取板块成分股列表
    @staticmethod
    def get_stock_list_in_sector(sector_name="沪深A股"):
        # sector_name="沪深A股"
        # sector_name="中证500"
        # sector_name="中证1000"
        # sector_name="中证2000"
        # sector_name="沪深300"
        # sector_name="上证50"
        # sector_name="SW1钢铁"
        ids=xtdata.get_stock_list_in_sector(sector_name)
        return ids

    ####订阅单股行情,单股订阅数量不宜过多，详见 接口概述-请求限制,单股订阅数量不超过50
    @staticmethod
    def  subscribe_quote_single(ids=['601288.SH','000001.SZ'],period='1d'):    
        xtdata.subscribe_quote(ids, period=period, start_time='', end_time='', count=0, callback=None)


    ####全推数据是市场全部合约的切面数据，是高订阅数场景下的有效解决方案。持续订阅全推数据可以获取到每个合约最新分笔数据的推送，且流量和处理效率都优于单股订阅
    @staticmethod
    def  subscribe_quote_whole(ids=['SH', 'SZ'],period='1d'):  
        # 传入市场代码代表订阅全市场，示例：['SH', 'SZ']
        # 传入合约代码代表订阅指定的合约，示例：['600000.SH', '000001.SZ']
        def callback(datas):
            print(datas)
            # global cc
            # cc=datas
            # for id in datas:
            #     print(id, datas[id])

        xtdata.subscribe_whole_quote(ids, callback=None)
        xtdata.run()
        
        
    ####从缓存获取行情数据，是主动获取行情的主要接口
    @staticmethod
    def  get_market_data(ids=['600000.SH', '000001.SZ'],period='1d'):         
        data=xtdata.get_market_data(field_list=[], stock_list=ids, period='1d', start_time='', end_time='', count=-1, dividend_type='none', fill_data=True)
        return data
        
    ####获取当前主力合约
    @staticmethod
    def  get_main_contract(id="IC00.IF"): 
        # https://dict.thinktrader.net/dictionary/future.html#%E4%B8%AD%E8%AF%81-1000-%E8%82%A1%E6%8C%87%E6%9C%9F%E8%B4%A7%E5%90%88%E7%BA%A6%E8%A1%A8
        # IF是中金所
        # "IH00.IF" 中金所的上证 50 指数期货
        # "IF00.IF" 中金所的沪深 300 指数期货
        # "IC00.IF" 中金所的中证 500 指数期货
        # "IM00.IF" 中金所的中证 1000 指数期货
        id_main=xtdata.get_main_contract(id) # 返回的主力合约代码
        return id_main

        
    ####传入指数代码，返回对应的期货合约（当前）
    @staticmethod
    def get_financial_futures_code_from_index(index_code:str) -> list:
        """
        Args:
            index_code:指数代码，如"000300.SH","000905.SH"
        Retuen:
            list: 对应期货合约列表
        """
        financial_futures = xtdata.get_stock_list_in_sector("中金所")
        future_list = []
        pattern = r'^[a-zA-Z]{1,2}\d{3,4}\.[A-Z]{2}$'
        for i in financial_futures:
            
            if re.match(pattern,i):
                future_list.append(i)
        ls = []
        for i in future_list:
            _info = xtdata._get_instrument_detail(i)
            _index_code = _info["ExtendInfo"]['OptUndlCode'] + "." + _info["ExtendInfo"]['OptUndlMarket']
            if _index_code == index_code:
                ls.append(i)
        return ls
    
    ####一旦被订阅股票的tick数据发生变动，除法回调函数，实时监控行情变化
    @staticmethod
    def  get_main_contract(ids=['601288.SH','000001.SZ'],period='1d'): 
        # 如果不想用固定间隔触发，可以以用订阅后的回调来执行
        # 这种模式下当订阅的callback回调函数将会异步的执行，每当订阅的标的tick发生变化更新，callback回调函数就会被调用一次
        # 本地已有的数据不会触发callback
        # 定义的回测函数
        ## 回调函数中，data是本次触发回调的数据，只有一条
        def callback(data):
            code_list = list(data.keys())    # 获取到本次触发的标的代码
            print(code_list)
            kline_in_callabck = xtdata.get_market_data_ex([],code_list,period = period)    # 在回调中获取klines数据
            print(list(kline_in_callabck.values())[0].iloc[-1])
        
        for id in ids:
            xtdata.subscribe_quote(id,period=period,count=-1,callback=callback) # 订阅时设定回调函数
        
        # 使用回调时，必须要同时使用xtdata.run()来阻塞程序，否则程序运行到最后一行就直接结束退出了。
        xtdata.run()


    @staticmethod
    def  trade_by_pos_change_fake(args,
                                   date,
                                   id,
                                   pre_pos, 
                                   target_pos):
        
        data = xtdata.get_market_data_ex([],[id],period='1d',count=-1)
        lastPrice=data[id].loc[date.replace('-',''),'close'] ##查询该id不复权指定天收盘价
        # lastPrice=data[id].loc[date.replace('-',''),'open'] ##查询该id不复权指定天开盘价
        trade_num = trade_price = trade_direction = pos_value = np.nan ##默认值

        if lastPrice==0:
            print(f'{id} lastPrice=0,可能存在停牌情况，请检查')
            return trade_num, trade_price, trade_direction, pos_value
        
        if target_pos>pre_pos:
            cash=round(args.money_stock*(target_pos-pre_pos)*2,2)
            trade_price=round(lastPrice*(1+args.slippage),2)
            trade_num=int(cash//(trade_price*100)*100)
            trade_direction='买入'    
            print(f'##############{id} buy {trade_num} on price {trade_price}')
        
        elif target_pos<pre_pos:
            cash=round(args.money_stock*(pre_pos-target_pos)*2,2)
            trade_price=round(lastPrice*(1-args.slippage),2)
            trade_num=int(cash//(trade_price*100)*100)
            trade_direction='卖出'                
            print(f'##############{id} buy {trade_num} on price {trade_price}')
        
        elif target_pos==pre_pos:
            pass
        pos_value=args.money_stock*target_pos*2 ##持仓金额=总资产*持仓权重
        
        return trade_num, trade_price, trade_direction, pos_value


    @staticmethod
    def  trade_by_pos_change_enhance_fake(args,
                                       date,
                                       id,
                                       pre_pos, 
                                       target_pos):
        
        data = xtdata.get_market_data_ex([],[id],period='1d',count=-1)
        lastPrice=data[id].loc[date.replace('-',''),'close'] ##查询该id不复权指定天收盘价
        # lastPrice=data[id].loc[date.replace('-',''),'open'] ##查询该id不复权指定天开盘价
        trade_num = trade_price = trade_direction = pos_value = np.nan ##默认值

        if lastPrice==0:
            print(f'{id} lastPrice=0,可能存在停牌情况，请检查')
            return trade_num, trade_price, trade_direction, pos_value
        
        if target_pos>pre_pos:
            cash=round(args.money_stock*(target_pos-pre_pos),2)
            trade_price=round(lastPrice*(1+args.slippage),2)
            trade_num=int(cash//(trade_price*100)*100)
            trade_direction='买入'    
            print(f'##############{id} buy {trade_num} on price {trade_price}')
        
        elif target_pos<pre_pos:
            cash=round(args.money_stock*(pre_pos-target_pos),2)
            trade_price=round(lastPrice*(1-args.slippage),2)
            trade_num=int(cash//(trade_price*100)*100)
            trade_direction='卖出'                
            print(f'##############{id} buy {trade_num} on price {trade_price}')
        
        elif target_pos==pre_pos:
            pass
        pos_value=args.money_stock*target_pos ##持仓金额=总资产*持仓权重
        
        return trade_num, trade_price, trade_direction, pos_value
    @staticmethod
    def  clean_all_pos(): ##全部清仓
        path = 'D:/project/quant/software/QMT_trail_version/userdata_mini'
        # session_id为会话编号，策略使用方对于不同的Python策略需要使用不同的会话编号
        session_id = 123
        from miniqmt import MyXtQuantTraderCallback
        xt_trader = XtQuantTrader(path, session_id)
        # acc = StockAccount('00617869') ## 密码 57608199
        # acc = StockAccount('2004216') ## 密码 123456
        acc = StockAccount('2004219') ## 密码 123456
        # acc = StockAccount('1000000365','STOCK')
        # 创建交易回调类对象，并声明接收回调
        callback = MyXtQuantTraderCallback()
        xt_trader.register_callback(callback)
        # 启动交易线程
        xt_trader.start()
        # 建立交易连接，返回0表示连接成功
        connect_result = xt_trader.connect()
        if connect_result==0:
            print('miniqmt连接成功')
        else:
            print('miniqmt连接失败')
        # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
        subscribe_result = xt_trader.subscribe(acc)
        if subscribe_result==0:
            print('账户订阅成功')
        else:
            print('账户订阅失败')
        # 查询当日所有的持仓
        ids_pos = xt_trader.query_stock_positions(acc)   
        res=[]
        for id_pos in ids_pos:
            res.append([id_pos.stock_code,id_pos.can_use_volume,id_pos.volume,id_pos.market_value])    
        res=pd.DataFrame(res,columns=['id','can_use_volume','volume','market_value']) # 查询当日所有的持仓  
        ids=list(res['id'])
        
        def callback(data):
            print(data)
            
        subscribe_id=xtdata.subscribe_whole_quote(ids, callback=None) ##订阅关联股票的实时数据
        tick=xtdata.get_full_tick(ids)
        
        for i in range(len(res)):
            id=res['id'].iloc[i]
            can_use_volume=res['can_use_volume'].iloc[i]
            if can_use_volume:
                lastPrice=xtdata.get_full_tick([id])[id]['lastPrice']
                print(f'##############{id} sell {can_use_volume} on price {round(lastPrice*(1-0.03),2)}')
                xt_trader.order_stock_async(acc, id, xtconstant.STOCK_SELL, int(can_use_volume), xtconstant.PRTP_FIX, round(lastPrice*(1-0.03),2), 'clean_all_pos', 'V0')                 
        
        xtdata.unsubscribe_quote(subscribe_id)
        
        
# ####期权api
# from xtquant import xtdata
# data=xtdata.get_option_undl_data('510500.SH') ##获取指定期权标的对应的期权品种列表,能查出'10005812.SHO'期权代码
# data=xtdata.get_option_list('510500.SH',dedate=None,opttype=None,isavailable=None) ##获取历史期权列表
# data=xtdata.get_option_detail_data('10005812.SHO') ##查询期权基础信息

# ids=['10005812.SHO']
# period='1d'
# def callback(data):
#     print(data)
# ##download_history_data2这个api的start_time有问题，下载数据的起始时间对不上
# xtdata.download_history_data2(ids, period=period, start_time='', end_time='', callback=callback,incrementally=True)  
# subscribe_id=xtdata.subscribe_whole_quote(ids, callback=None) ##订阅关联股票的实时数据
# time.sleep(1)  ##等待订阅完成
# tick=xtdata.get_full_tick(ids)
# data=xtdata.get_market_data_ex(field_list = [], stock_list = ids, period = '1d', start_time = '', end_time = '', count = -1, dividend_type = 'back_ratio', fill_data = True)



if __name__ == '__main__':
    data=xtdata.get_option_undl_data('510500.SH')

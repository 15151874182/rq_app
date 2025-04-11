import os,sys,time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import traceback
from copy import deepcopy 
import gc

import rqdatac
from tools.convert_func import Convert  
from tools.metrics_func import Metrics
from tools.general_func import General
from tools.plot_func import Plot
rqdatac.init()

from rqalpha_plus import *
from rqalpha.apis import *
import rqalpha
import rqalpha_mod_fund

__config__ = {
    "base": {
        "accounts": {
            "STOCK": 6000000,
        },
        "start_date": "20240101",
        "end_date": "20251231",
    },
    

    # "sys_simulation": {
    #     "price_limit": False
    # },
    
    
    "mod": {
        "sys_analyser": {
            "plot": True,
            "benchmark": "932000.INDX"
        }
    }
}

def read_tables_df():
    # need  pandas version 0.21.0+
    # need xlrd
    d_type = {'NAME': str, 'TARGET_WEIGHT': float, 'TICKER': str, 'TRADE_DT': int}
    columns_name = ["TRADE_DT", "TICKER", "NAME", "TARGET_WEIGHT"]
    df = pd.read_excel(r'调仓权重2.xlsx', dtype=d_type)
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
    order_book_id = event.order.order_book_id
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
        for _ticker in context.target_queue:
            flag=is_suspended_df.loc[context.now.strftime('%Y-%m-%d'),_ticker]
            if flag:
                continue
        
            _target_weight = context.current_target_table.get(_ticker, 0)
            # o = order_target_percent(_ticker, round(_target_weight, 6))
            o = order_shares(_ticker, 200)
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


if __name__ == '__main__':
    ### 调仓权重2.xlsx文件的制作
    inputs=[]
    st='20240101'
    et='20250322'
    dates=rqdatac.get_trading_dates(st, et, market='cn')
    is_suspendeds=[]
    for date in tqdm(dates):
        weights=rqdatac.index_weights(order_book_id='866006.RI', date=date)
        weights=weights.reset_index()
        weights.columns=['TICKER','TARGET_WEIGHT']
        weights['TRADE_DT']=date.strftime('%Y%m%d')
        weights['NAME']=weights['TICKER'].apply(lambda id:rqdatac.instruments(id, market='cn').symbol)
        weights=weights[['TRADE_DT','TICKER','NAME','TARGET_WEIGHT']]
        is_suspendeds+=list(weights['TICKER'])
        inputs.append(weights)
    
    is_suspendeds_df=pd.DataFrame(set(is_suspendeds),columns=['id'])
    is_suspended_df=rqdatac.is_suspended(list(is_suspendeds_df['id']), start_date=st,end_date=et)
    
    # inputs=pd.concat(inputs,axis=0)
    
    # with pd.ExcelWriter('调仓权重2.xlsx', engine='xlsxwriter') as writer:
    #     inputs.to_excel(writer, sheet_name='', index=False)  
        
    res=run_func(init=init, before_trading=before_trading, after_trading=after_trading, handle_bar=handle_bar,
              config=__config__)
    xx=1
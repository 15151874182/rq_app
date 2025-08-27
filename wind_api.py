# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:54:44 2025

@author: 陈天宇
"""
import rqdatac
rqdatac.init()
import pandas as pd

dates=rqdatac.get_trading_dates(start_date='20170101', end_date='20250825')
fridays = [d for d in dates if d.weekday() == 4]
fridays=pd.DataFrame(fridays)
from WindPy import w
w.start()





for date in ['2025-08-24','2025-08-25']:
    xx=w.wset(f"sectorconstituent","date={date};windcode=8841090.WI")
    cc=pd.DataFrame(xx.Data).T
    cc.columns=['date','id','name']
    cc.to_csv(f'D:/cty/quant/wind_csv/yz_index/8841090.WI-{date}.csv')